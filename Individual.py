import numpy as np
from typing import Optional, Tuple
from EvolutionPolicy import EvolutionPolicy

class Individual:
    """
    Jedinec genetického algoritmu:
      - drží policy (MLP) + fitness
      - vie sa porovnať podľa fitness
      - má mutate a crossover
    """
    def __init__(self, obs_dim: int, hidden_dim: int, act_dim: int,
                 genome: Optional[np.ndarray] = None) -> None:
        self.policy = EvolutionPolicy(obs_dim, hidden_dim, act_dim, genome)
        self.fitness: Optional[float] = None  # zatiaľ nevyhodnotený

    @property
    def genome(self) -> np.ndarray:
        return self.policy.genome

    @genome.setter
    def genome(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32)
        if value.shape[0] != self.policy.genome_size:
            raise ValueError(
                f"Nový genóm má dĺžku {value.shape[0]}, ale očakávané {self.policy.genome_size}"
            )
        self.policy.genome = value
        self.fitness = None  # po zmene váh treba fitness znova spočítať

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.policy.act(obs)

    # ------- porovnávanie jedincov podľa fitness -------

    def _fitness_value(self) -> float:
        """
        Interné: ak fitness ešte nie je spočítané, berieme -inf,
        aby sa nevyhodnotený jedinec bral ako najhorší.
        """
        if self.fitness is None:
            return float("-inf")
        return float(self.fitness)

    def __lt__(self, other: "Individual") -> bool:
        return self._fitness_value() < other._fitness_value()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self._fitness_value() == other._fitness_value()

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness})"

    # ------- genetické operácie -------

    def copy(self) -> "Individual":
        """
        Hlboká kópia jedinca (genóm sa kopíruje).
        """
        new = Individual(
            obs_dim=self.policy.obs_dim,
            hidden_dim=self.policy.hidden_dim,
            act_dim=self.policy.act_dim,
            genome=self.genome.copy(),
        )
        new.fitness = self.fitness
        return new

    def mutate(self, mutation_prob: float = 0.1, sigma: float = 0.1) -> None:
        """
        Jednoduchá mutácia: s pravdepodobnosťou mutation_prob
        pridáme k danej váhe gaussovský šum ~ N(0, sigma^2).
        """
        g = self.genome
        mask = np.random.rand(g.size) < mutation_prob
        g[mask] += np.random.randn(mask.sum()).astype(np.float32) * sigma
        self.fitness = None

    def crossover(self, other: "Individual") -> "Individual":
        """
        Jednobodové kríženie: vracia jedného potomka,
        ktorého genóm je kombinácia rodičov.
        """
        if self.genome.shape != other.genome.shape:
            raise ValueError("Jedinci majú rozdielnu veľkosť genómu.")

        point = np.random.randint(1, self.genome.size)  # 0 < point < size
        child_genome = np.empty_like(self.genome)
        child_genome[:point] = self.genome[:point]
        child_genome[point:] = other.genome[point:]

        return Individual(
            obs_dim=self.policy.obs_dim,
            hidden_dim=self.policy.hidden_dim,
            act_dim=self.policy.act_dim,
            genome=child_genome,
        )