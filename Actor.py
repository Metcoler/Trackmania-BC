import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np

from Car import Car
from Map import Map
from ObservationEncoder import ObservationEncoder
from XboxController import XboxControllerReader, XboxControllerState


@dataclass
class AttemptSample:
    observation: np.ndarray
    action: np.ndarray
    game_time: float
    total_progress: float
    distance: float


class AttemptWriter:
    SUMMARY_HEADERS = [
        "attempt_index",
        "saved",
        "num_frames",
        "finish_time",
        "total_progress",
        "distance",
        "path",
    ]

    def __init__(self, base_dir: str, map_name: str, encoder: ObservationEncoder) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_map_{map_name}_target_dataset"
        self.run_dir = os.path.join(base_dir, run_name)
        self.attempts_dir = os.path.join(self.run_dir, "attempts")
        os.makedirs(self.attempts_dir, exist_ok=False)
        self.summary_path = os.path.join(self.run_dir, "attempts.csv")
        with open(self.summary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.SUMMARY_HEADERS)
            writer.writeheader()

        config = {
            "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "map_name": map_name,
            "observation_dim": encoder.obs_dim,
            "dt_ref": encoder.dt_ref,
            "dt_ratio_clip": encoder.dt_ratio_clip,
            "action_mode": "target",
            "action_layout": ["gas", "brake", "steer"],
        }
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=True)

    def save_attempt(
        self,
        attempt_index: int,
        samples: List[AttemptSample],
        finish_info: Dict[str, float],
    ) -> str:
        if not samples:
            raise ValueError("Cannot save an empty attempt.")

        observations = np.stack([sample.observation for sample in samples]).astype(np.float32)
        actions = np.stack([sample.action for sample in samples]).astype(np.float32)
        game_times = np.array([sample.game_time for sample in samples], dtype=np.float32)
        total_progress = np.array([sample.total_progress for sample in samples], dtype=np.float32)
        distances = np.array([sample.distance for sample in samples], dtype=np.float32)

        output_path = os.path.join(self.attempts_dir, f"attempt_{attempt_index:04d}.npz")
        np.savez(
            output_path,
            observations=observations,
            actions=actions,
            game_times=game_times,
            total_progress=total_progress,
            distances=distances,
            finish_time=np.array([float(finish_info.get("time", 0.0))], dtype=np.float32),
            finish_progress=np.array([float(finish_info.get("total_progress", 0.0))], dtype=np.float32),
            finish_distance=np.array([float(finish_info.get("distance", 0.0))], dtype=np.float32),
        )
        self._append_summary(
            dict(
                attempt_index=attempt_index,
                saved=1,
                num_frames=len(samples),
                finish_time=float(finish_info.get("time", 0.0)),
                total_progress=float(finish_info.get("total_progress", 0.0)),
                distance=float(finish_info.get("distance", 0.0)),
                path=output_path,
            )
        )
        return output_path

    def log_discard(self, attempt_index: int, samples: List[AttemptSample], finish_info: Dict[str, float]) -> None:
        self._append_summary(
            dict(
                attempt_index=attempt_index,
                saved=0,
                num_frames=len(samples),
                finish_time=float(finish_info.get("time", 0.0)),
                total_progress=float(finish_info.get("total_progress", 0.0)),
                distance=float(finish_info.get("distance", 0.0)),
                path="",
            )
        )

    def _append_summary(self, row: Dict) -> None:
        with open(self.summary_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.SUMMARY_HEADERS)
            writer.writerow(row)


def controller_to_action(state: XboxControllerState) -> np.ndarray:
    return np.array([state.gas, state.brake, state.steer], dtype=np.float32)


def rising_edge(current: int, previous: int) -> bool:
    return bool(current) and not bool(previous)


if __name__ == "__main__":
    map_name = "small_map"
    base_dir = "logs/supervised_data"
    encoder = ObservationEncoder(dt_ref=1.0 / 100.0, dt_ratio_clip=3.0)
    writer = AttemptWriter(base_dir=base_dir, map_name=map_name, encoder=encoder)

    print(f"Saving supervised attempts into: {writer.run_dir}")
    print("Workflow:")
    print("- recording starts when game time becomes > 0")
    print("- press B during an attempt to discard it")
    print("- after finish, press A to save or B to discard")
    print("- stop script with Ctrl+C")

    game_map = Map(map_name)
    car = Car(game_map)
    controller = XboxControllerReader()

    attempt_index = 1
    state = "waiting_for_start"
    attempt_samples: List[AttemptSample] = []
    previous_action = np.zeros(3, dtype=np.float32)
    last_buttons = {"a": 0, "b": 0}
    finish_info: Dict[str, float] = {}

    try:
        while True:
            distances, instructions, info = car.get_data()
            snapshot = controller.snapshot()
            action = controller_to_action(snapshot)
            a_pressed = rising_edge(snapshot.button_a, last_buttons["a"])
            b_pressed = rising_edge(snapshot.button_b, last_buttons["b"])
            last_buttons["a"] = snapshot.button_a
            last_buttons["b"] = snapshot.button_b

            game_time = float(info.get("time", 0.0))
            total_progress = float(info.get("total_progress", 0.0))
            total_distance = float(info.get("distance", 0.0))
            finished = bool(info.get("done", 0.0) == 1.0)

            if state == "waiting_for_start":
                if game_time > 0.0:
                    encoder.reset()
                    attempt_samples = []
                    previous_action = np.zeros(3, dtype=np.float32)
                    state = "recording"
                    print(f"Attempt {attempt_index:04d} started.")

            if state == "recording":
                observation = encoder.build_observation(
                    distances=distances,
                    instructions=instructions,
                    info=info,
                    previous_action=previous_action,
                )
                attempt_samples.append(
                    AttemptSample(
                        observation=observation,
                        action=action.copy(),
                        game_time=game_time,
                        total_progress=total_progress,
                        distance=total_distance,
                    )
                )
                previous_action = action.copy()

                if b_pressed:
                    print(f"Attempt {attempt_index:04d} discarded by B restart.")
                    writer.log_discard(attempt_index, attempt_samples, dict(time=game_time, total_progress=total_progress, distance=total_distance))
                    attempt_index += 1
                    attempt_samples = []
                    state = "waiting_for_reset"
                elif finished:
                    finish_info = dict(time=game_time, total_progress=total_progress, distance=total_distance)
                    print(
                        f"Attempt {attempt_index:04d} finished in {game_time:.2f}s. "
                        "Press A to save or B to discard."
                    )
                    state = "await_finish_confirmation"
                elif game_time <= 0.0:
                    print(f"Attempt {attempt_index:04d} reset before finish. Discarded.")
                    writer.log_discard(attempt_index, attempt_samples, dict(time=game_time, total_progress=total_progress, distance=total_distance))
                    attempt_index += 1
                    attempt_samples = []
                    state = "waiting_for_start"

            elif state == "await_finish_confirmation":
                if a_pressed:
                    output_path = writer.save_attempt(attempt_index, attempt_samples, finish_info)
                    print(f"Attempt {attempt_index:04d} saved: {output_path}")
                    attempt_index += 1
                    attempt_samples = []
                    state = "waiting_for_reset"
                elif b_pressed:
                    print(f"Attempt {attempt_index:04d} discarded after finish.")
                    writer.log_discard(attempt_index, attempt_samples, finish_info)
                    attempt_index += 1
                    attempt_samples = []
                    state = "waiting_for_reset"

            elif state == "waiting_for_reset":
                if game_time <= 0.0:
                    state = "waiting_for_start"

    except KeyboardInterrupt:
        print("\nStopped data collection.")
    finally:
        controller.close()
