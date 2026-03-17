import glob
import json
import os
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from EvolutionPolicy import EvolutionPolicy
from ObservationEncoder import ObservationEncoder


def find_attempt_files(base_dir: str = "logs/supervised_data") -> List[str]:
    pattern = os.path.join(base_dir, "**", "attempts", "attempt_*.npz")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No attempt files found under {base_dir}.")
    return sorted(files)


def load_attempt_dataset(
    attempt_files: List[str],
    boring_keep_probability: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    observations = []
    actions = []
    total_frames = 0

    for path in attempt_files:
        with np.load(path) as data:
            obs = np.asarray(data["observations"], dtype=np.float32)
            act = np.asarray(data["actions"], dtype=np.float32)
            observations.append(obs)
            actions.append(act)
            total_frames += int(obs.shape[0])

    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)

    boring_mask = (
        (actions[:, 0] > 0.95)
        & (actions[:, 1] < 0.05)
        & (np.abs(actions[:, 2]) < 0.05)
    )
    boring_count = int(np.count_nonzero(boring_mask))
    keep_mask = (~boring_mask) | (rng.random(actions.shape[0]) < boring_keep_probability)
    observations = observations[keep_mask]
    actions = actions[keep_mask]
    kept_after_filter = int(observations.shape[0])

    mirrored_observations = np.stack(
        [ObservationEncoder.mirror_observation(obs) for obs in observations], axis=0
    ).astype(np.float32)
    mirrored_actions = actions.copy()
    mirrored_actions[:, 2] *= -1.0

    observations = np.concatenate([observations, mirrored_observations], axis=0)
    actions = np.concatenate([actions, mirrored_actions], axis=0)

    permutation = rng.permutation(observations.shape[0])
    observations = observations[permutation]
    actions = actions[permutation]

    stats = {
        "attempt_files": len(attempt_files),
        "total_frames_before_filter": total_frames,
        "boring_frames_before_filter": boring_count,
        "frames_after_filter_before_mirror": kept_after_filter,
        "frames_after_mirror": int(observations.shape[0]),
    }
    return observations, actions, stats


if __name__ == "__main__":
    data_root = "logs/supervised_data"
    output_root = "logs/supervised_runs"
    hidden_dims = (128, 64)
    hidden_activation = "relu"
    batch_size = 1024
    epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-5
    validation_fraction = 0.1
    boring_keep_probability = 0.1
    random_seed = 42

    rng = np.random.default_rng(random_seed)
    attempt_files = find_attempt_files(data_root)
    observations, actions, dataset_stats = load_attempt_dataset(
        attempt_files=attempt_files,
        boring_keep_probability=boring_keep_probability,
        rng=rng,
    )

    obs_dim = int(observations.shape[1])
    act_dim = int(actions.shape[1])
    split_index = max(1, int(observations.shape[0] * (1.0 - validation_fraction)))
    train_obs = observations[:split_index]
    train_actions = actions[:split_index]
    val_obs = observations[split_index:]
    val_actions = actions[split_index:]
    if val_obs.shape[0] == 0:
        val_obs = train_obs[-1:].copy()
        val_actions = train_actions[-1:].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EvolutionPolicy(
        obs_dim=obs_dim,
        hidden_dim=hidden_dims,
        act_dim=act_dim,
        action_mode="target",
        hidden_activation=hidden_activation,
        action_scale=np.ones(act_dim, dtype=np.float32),
        device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_weights = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32, device=device)
    criterion = nn.SmoothL1Loss(reduction="none")

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_obs),
            torch.from_numpy(train_actions),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(val_obs),
            torch.from_numpy(val_actions),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_target_supervised"
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=False)
    best_model_path = os.path.join(run_dir, "best_model.pt")
    curve_path = os.path.join(run_dir, "loss_curve.png")

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_count = 0
        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_obs)
            per_value_loss = criterion(predictions, batch_actions)
            loss = (per_value_loss * loss_weights).mean()
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_obs.shape[0]
            running_train_loss += float(loss.detach().cpu()) * batch_size_actual
            train_count += batch_size_actual

        train_loss = running_train_loss / max(train_count, 1)
        train_losses.append(train_loss)

        model.eval()
        running_val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_obs, batch_actions in val_loader:
                batch_obs = batch_obs.to(device)
                batch_actions = batch_actions.to(device)
                predictions = model(batch_obs)
                per_value_loss = criterion(predictions, batch_actions)
                loss = (per_value_loss * loss_weights).mean()

                batch_size_actual = batch_obs.shape[0]
                running_val_loss += float(loss.detach().cpu()) * batch_size_actual
                val_count += batch_size_actual

        val_loss = running_val_loss / max(val_count, 1)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            model.save(
                best_model_path,
                extra={
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "dataset_stats": dataset_stats,
                },
            )

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Supervised Target Policy Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()

    summary = {
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_root": data_root,
        "attempt_files": attempt_files,
        "dataset_stats": dataset_stats,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden_dims": list(hidden_dims),
        "hidden_activation": hidden_activation,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "validation_fraction": validation_fraction,
        "boring_keep_probability": boring_keep_probability,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_model_path": best_model_path,
        "curve_path": curve_path,
        "device": str(device),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(f"Best model saved to: {best_model_path}")
    print(f"Training summary saved to: {os.path.join(run_dir, 'summary.json')}")
