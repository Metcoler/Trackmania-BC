import numpy as np
from typing import Optional, Tuple


class EvolutionPolicy:
    """
    Jednoduchá MLP politika s jednou skrytou vrstvou:
        obs_dim -> hidden_dim -> act_dim

    Všetky váhy sú uložené v jednom 1D genóme (vektore).
    """
    def __init__(self, obs_dim: int, hidden_dim: int, act_dim: int,
                 genome: Optional[np.ndarray] = None) -> None:
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim

        self.genome_size = self.compute_genome_size(obs_dim, hidden_dim, act_dim)

        if genome is None:
            # náhodná inicializácia okolo 0
            self.genome = np.random.randn(self.genome_size).astype(np.float32) * 0.1
        else:
            genome = np.asarray(genome, dtype=np.float32)
            if genome.shape[0] != self.genome_size:
                raise ValueError(
                    f"Genome má dĺžku {genome.shape[0]}, očakávané {self.genome_size}"
                )
            self.genome = genome

    @staticmethod
    def compute_genome_size(obs_dim: int, hidden_dim: int, act_dim: int) -> int:
        """
        hidden vrstva:  hidden_dim x (obs_dim + 1)  (+1 je bias)
        výstupná vrstva: act_dim x (hidden_dim + 1)
        """
        w1_size = hidden_dim * (obs_dim + 1)
        w2_size = act_dim * (hidden_dim + 1)
        return w1_size + w2_size

    def _decode_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rozbalí 1D genóm do matíc W1, W2.
        W1: (hidden_dim, obs_dim + 1)
        W2: (act_dim, hidden_dim + 1)
        """
        w1_size = self.hidden_dim * (self.obs_dim + 1)
        w1_flat = self.genome[:w1_size]
        w2_flat = self.genome[w1_size:]

        W1 = w1_flat.reshape(self.hidden_dim, self.obs_dim + 1)
        W2 = w2_flat.reshape(self.act_dim, self.hidden_dim + 1)
        return W1, W2

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Forward pass:
            x -> tanh(W1 * [x;1]) -> tanh(W2 * [h;1]) -> škálovanie do [-0.2, 0.2]
        """
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 1:
            raise ValueError(f"Očakávaný 1D observation vektor, dostal som tvar {obs.shape}")

        if obs.shape[0] != self.obs_dim:
            # môžeš sem dať assert, alebo len warning, podľa toho ako to chceš striktne
            raise ValueError(
                f"Obs dim {obs.shape[0]} nesedí s očakávaným {self.obs_dim}"
            )

        W1, W2 = self._decode_weights()

        # vstup + bias
        x = np.concatenate([obs, np.array([1.0], dtype=np.float32)])  # (obs_dim + 1,)
        h = np.tanh(W1 @ x)  # (hidden_dim,)

        # skrytá vrstva + bias
        h_with_bias = np.concatenate([h, np.array([1.0], dtype=np.float32)])  # (hidden_dim + 1,)
        y_raw = np.tanh(W2 @ h_with_bias)  # (act_dim,)

        # škálovanie do [-0.2, 0.2]
        action = 0.2 * y_raw
        return action.astype(np.float32)
