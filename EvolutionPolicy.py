from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


HiddenDims = Union[int, Sequence[int]]
HiddenActivations = Union[str, Sequence[str]]


def normalize_hidden_dims(hidden_dim: HiddenDims) -> Tuple[int, ...]:
    if isinstance(hidden_dim, (tuple, list)):
        dims = tuple(int(dim) for dim in hidden_dim)
    else:
        dims = (int(hidden_dim),)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("hidden_dim must contain positive integers.")
    return dims


def canonicalize_activation_name(name: str) -> str:
    activation = str(name).strip().lower()
    if activation == "tann":
        activation = "tanh"
    if activation not in {"tanh", "relu", "sigmoid"}:
        raise ValueError(
            f"Unsupported hidden activation '{name}'. "
            "Use 'tanh', 'relu', or 'sigmoid'."
        )
    return activation


def normalize_hidden_activations(
    hidden_activation: HiddenActivations,
    num_hidden_layers: int,
) -> Tuple[str, ...]:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive.")

    if isinstance(hidden_activation, str):
        activations = [hidden_activation]
    else:
        activations = list(hidden_activation)

    if not activations:
        raise ValueError("hidden_activation must contain at least one activation name.")

    normalized = tuple(canonicalize_activation_name(value) for value in activations)
    if len(normalized) == 1 and num_hidden_layers > 1:
        normalized = normalized * num_hidden_layers
    if len(normalized) != num_hidden_layers:
        raise ValueError(
            "hidden_activation must provide either one activation or exactly "
            f"{num_hidden_layers} activations, got {len(normalized)}."
        )
    return normalized


class EvolutionPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: HiddenDims,
        act_dim: int,
        genome: Optional[np.ndarray] = None,
        action_scale: Optional[np.ndarray] = None,
        action_mode: str = "delta",
        hidden_activation: HiddenActivations = "tanh",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dims = normalize_hidden_dims(hidden_dim)
        self.hidden_dim = self.hidden_dims[0] if len(self.hidden_dims) == 1 else self.hidden_dims
        self.act_dim = int(act_dim)
        self.action_mode = str(action_mode).strip().lower()
        if self.action_mode not in {"delta", "target"}:
            raise ValueError("action_mode must be 'delta' or 'target'.")
        self.hidden_activations = normalize_hidden_activations(
            hidden_activation=hidden_activation,
            num_hidden_layers=len(self.hidden_dims),
        )
        self.hidden_activation = (
            self.hidden_activations[0]
            if len(self.hidden_activations) == 1
            else list(self.hidden_activations)
        )
        if device is None:
            device = torch.device("cpu")
        self.device = torch.device(device)

        self.model = self._build_model()
        scale = self._normalize_action_scale(action_scale)
        self.register_buffer("action_scale", torch.tensor(scale, dtype=torch.float32))
        self.reset_parameters()
        self.to(self.device)

        if genome is not None:
            self.set_genome(genome)

    def _normalize_action_scale(self, action_scale: Optional[np.ndarray]) -> np.ndarray:
        if action_scale is None:
            return np.full((self.act_dim,), 0.2, dtype=np.float32)

        scale = np.asarray(action_scale, dtype=np.float32).reshape(-1)
        if scale.size == 1:
            scale = np.repeat(scale, self.act_dim).astype(np.float32)
        if scale.size != self.act_dim:
            raise ValueError(
                f"action_scale has size {scale.size}, expected 1 or {self.act_dim}."
            )
        return scale.astype(np.float32, copy=False)

    def _make_activation(self, activation_name: str) -> nn.Module:
        if activation_name == "relu":
            return nn.ReLU()
        if activation_name == "sigmoid":
            return nn.Sigmoid()
        if activation_name == "tanh":
            return nn.Tanh()
        raise ValueError(
            f"Unsupported hidden activation '{activation_name}'. "
            "Use 'tanh', 'relu', or 'sigmoid'."
        )

    def _build_model(self) -> nn.Sequential:
        layers = []
        in_dim = self.obs_dim
        for hidden, activation_name in zip(self.hidden_dims, self.hidden_activations):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(self._make_activation(activation_name))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.act_dim))
        return nn.Sequential(*layers)

    def reset_parameters(self) -> None:
        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.normal_(module.bias, mean=0.0, std=0.1)

    @property
    def genome_size(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @staticmethod
    def compute_genome_size(obs_dim: int, hidden_dim: HiddenDims, act_dim: int) -> int:
        hidden_dims = normalize_hidden_dims(hidden_dim)
        in_dim = int(obs_dim)
        total_size = 0
        for hidden in hidden_dims:
            total_size += int(hidden) * (in_dim + 1)
            in_dim = int(hidden)
        total_size += int(act_dim) * (in_dim + 1)
        return total_size

    def get_config(self) -> Dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "hidden_dim": list(self.hidden_dims),
            "hidden_dims": list(self.hidden_dims),
            "act_dim": self.act_dim,
            "action_mode": self.action_mode,
            "hidden_activation": list(self.hidden_activations),
            "hidden_activations": list(self.hidden_activations),
            "action_scale": self.action_scale.detach().cpu().numpy().astype(np.float32).tolist(),
        }

    def raw_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def transform_outputs(self, raw: torch.Tensor) -> torch.Tensor:
        if self.action_mode == "delta":
            scale = self.action_scale.to(raw.device)
            while scale.ndim < raw.ndim:
                scale = scale.unsqueeze(0)
            return torch.tanh(raw) * scale

        gas = torch.sigmoid(raw[..., 0:1])
        brake = torch.sigmoid(raw[..., 1:2])
        steer = torch.tanh(raw[..., 2:3])
        return torch.cat([gas, brake, steer], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_forward(x)
        return self.transform_outputs(raw)

    def act(self, obs) -> np.ndarray:
        observation = np.asarray(obs, dtype=np.float32).reshape(-1)
        if observation.shape[0] != self.obs_dim:
            raise ValueError(
                f"Obs dim {observation.shape[0]} does not match expected {self.obs_dim}."
            )

        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(observation).to(self.device).unsqueeze(0)
            action = self.forward(tensor).squeeze(0)
        return action.detach().cpu().numpy().astype(np.float32)

    @property
    def genome(self) -> np.ndarray:
        flat_parts = []
        with torch.no_grad():
            for parameter in self.model.parameters():
                flat_parts.append(parameter.detach().cpu().reshape(-1))
        return torch.cat(flat_parts).numpy().astype(np.float32)

    @genome.setter
    def genome(self, genome: np.ndarray) -> None:
        self.set_genome(genome)

    def set_genome(self, genome: np.ndarray) -> None:
        flat = np.asarray(genome, dtype=np.float32).reshape(-1)
        if flat.size != self.genome_size:
            raise ValueError(
                f"Genome has size {flat.size}, expected {self.genome_size}."
            )

        tensor = torch.from_numpy(flat)
        offset = 0
        with torch.no_grad():
            for parameter in self.model.parameters():
                numel = parameter.numel()
                view = tensor[offset : offset + numel].view_as(parameter)
                parameter.copy_(view.to(parameter.device))
                offset += numel

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "config": self.get_config(),
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(
        cls,
        path: str,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Tuple["EvolutionPolicy", Dict[str, Any]]:
        if map_location is None:
            map_location = torch.device("cpu")
        payload = torch.load(path, map_location=map_location)
        config = dict(payload.get("config", {}))
        hidden_dim = config.get("hidden_dims", config.get("hidden_dim"))
        if isinstance(hidden_dim, list):
            hidden_dim = tuple(hidden_dim)
        hidden_activation = config.get(
            "hidden_activations",
            config.get("hidden_activation", "tanh"),
        )
        policy = cls(
            obs_dim=int(config["obs_dim"]),
            hidden_dim=hidden_dim,
            act_dim=int(config["act_dim"]),
            action_scale=np.asarray(config.get("action_scale", [0.2, 0.2, 0.2]), dtype=np.float32),
            action_mode=str(config.get("action_mode", "delta")),
            hidden_activation=hidden_activation,
            device=map_location,
        )
        policy.load_state_dict(payload["state_dict"])
        policy.to(policy.device)
        return policy, dict(payload.get("extra", {}))
