from dataclasses import dataclass
import threading
import time

import inputs


@dataclass
class XboxControllerState:
    gas: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    button_a: int = 0
    button_b: int = 0


class XboxControllerReader:
    def __init__(self, deadzone: float = 0.1) -> None:
        self.deadzone = float(deadzone)
        self._alive = True
        self._lock = threading.Lock()
        self._state = XboxControllerState()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._alive:
            try:
                events = inputs.get_gamepad()
            except Exception:
                time.sleep(0.05)
                continue

            with self._lock:
                for event in events:
                    if event.ev_type == "Absolute":
                        if event.code == "ABS_RZ":
                            self._state.gas = round(event.state / 255.0, 3)
                        elif event.code == "ABS_Z":
                            self._state.brake = round(event.state / 255.0, 3)
                        elif event.code == "ABS_X":
                            steer = event.state / 32768.0
                            if abs(steer) < self.deadzone:
                                steer = 0.0
                            self._state.steer = round(max(-1.0, min(1.0, steer)), 3)
                    elif event.ev_type == "Key":
                        if event.code == "BTN_SOUTH":
                            self._state.button_a = int(event.state)
                        elif event.code == "BTN_EAST":
                            self._state.button_b = int(event.state)

    def snapshot(self) -> XboxControllerState:
        with self._lock:
            return XboxControllerState(**self._state.__dict__)

    def close(self) -> None:
        self._alive = False

    def __enter__(self) -> "XboxControllerReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
