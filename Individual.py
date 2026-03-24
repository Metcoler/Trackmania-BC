import numpy as np
from typing import Optional, Tuple

from EvolutionPolicy import EvolutionPolicy


class Individual:
    """
    Genetic algorithm individual.

    Holds a policy network plus evaluation metrics used by lexicographic ranking.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        act_dim: int,
        genome: Optional[np.ndarray] = None,
        action_scale: Optional[np.ndarray] = None,
        action_mode: str = "delta",
        hidden_activation: str = "tanh",
    ) -> None:
        self.policy = EvolutionPolicy(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            act_dim=act_dim,
            genome=genome,
            action_scale=action_scale,
            action_mode=action_mode,
            hidden_activation=hidden_activation,
        )

        self.fitness: Optional[float] = None
        self.total_progress: float = 0.0
        self.time: float = float("inf")
        self.term: int = -999
        self.distance: float = 0.0

    @property
    def genome(self) -> np.ndarray:
        return self.policy.genome

    @genome.setter
    def genome(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape[0] != self.policy.genome_size:
            raise ValueError(
                f"New genome has size {value.shape[0]}, expected {self.policy.genome_size}."
            )
        self.policy.genome = value
        self.invalidate_evaluation()

    def invalidate_evaluation(self) -> None:
        self.fitness = None
        self.total_progress = 0.0
        self.time = float("inf")
        self.term = -999
        self.distance = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.policy.act(obs)

    def ranking_key(self) -> Tuple[int, float, int, float]:
        term = int(self.term)
        progress = float(self.total_progress)
        dist = float(self.distance)
        t = float(self.time)

        if np.isfinite(t):
            time_bucket = int(np.floor(t))
        else:
            time_bucket = 10**9

        if term <= 0:
            dist = 0
        
        return (term, progress, -time_bucket, -dist)

    def compute_scalar_fitness(self) -> float:
        term, progress, neg_time_bucket, neg_dist = self.ranking_key()
        time_bucket = -neg_time_bucket
        dist = -neg_dist

        a = 1_000_000_000.0
        b = 1_000_000.0
        c = 10_000.0
        d = 1.0
        return term * a + progress * b - time_bucket * c - dist * d

    def __lt__(self, other: "Individual") -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.ranking_key() < other.ranking_key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.ranking_key() == other.ranking_key()

    def __repr__(self) -> str:
        return (
            "Individual("
            f"term={self.term}, "
            f"progress={self.total_progress:.1f}, "
            f"distance={self.distance:.1f}, "
            f"time={self.time:.2f}, "
            f"fitness={self.fitness})"
        )

    def copy(self) -> "Individual":
        new = Individual(
            obs_dim=self.policy.obs_dim,
            hidden_dim=self.policy.hidden_dims,
            act_dim=self.policy.act_dim,
            genome=self.genome.copy(),
            action_scale=self.policy.action_scale.detach().cpu().numpy().copy(),
            action_mode=self.policy.action_mode,
            hidden_activation=self.policy.hidden_activation,
        )
        new.fitness = self.fitness
        new.total_progress = self.total_progress
        new.time = self.time
        new.term = self.term
        new.distance = self.distance
        return new

    def mutate(self, mutation_prob: float = 0.1, sigma: float = 0.1) -> None:
        genome = self.genome.copy()
        mask = np.random.rand(genome.size) < mutation_prob
        if np.any(mask):
            genome[mask] += np.random.randn(mask.sum()).astype(np.float32) * sigma
        self.genome = genome

    def crossover(self, other: "Individual") -> "Individual":
        if self.genome.shape != other.genome.shape:
            raise ValueError("Individuals have different genome sizes.")

        point = np.random.randint(1, self.genome.size)
        child_genome = np.empty_like(self.genome)
        child_genome[:point] = self.genome[:point]
        child_genome[point:] = other.genome[point:]

        return Individual(
            obs_dim=self.policy.obs_dim,
            hidden_dim=self.policy.hidden_dims,
            act_dim=self.policy.act_dim,
            genome=child_genome,
            action_scale=self.policy.action_scale.detach().cpu().numpy().copy(),
            action_mode=self.policy.action_mode,
            hidden_activation=self.policy.hidden_activation,
        )
