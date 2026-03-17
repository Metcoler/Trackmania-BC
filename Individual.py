import numpy as np
from typing import Optional, Tuple
from EvolutionPolicy import EvolutionPolicy


class Individual:
    """
    Jedinec genetického algoritmu:
      - drží policy (MLP)
      - multi-kritériá hodnotenia:
          * term            (<= -1 crash / touch-limit, 0 timeout, 1 finish)
          * total_progress  [0..100] % trate
          * distance        celková prejdená vzdialenosť (z OpenPlanet: data["distance"])
          * time            čas v sekundách
      - scalar fitness len na logovanie / históriu
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        act_dim: int,
        genome: Optional[np.ndarray] = None,
        action_scale: Optional[np.ndarray] = None,
    ) -> None:
        self.policy = EvolutionPolicy(
            obs_dim,
            hidden_dim,
            act_dim,
            genome,
            action_scale=action_scale,
        )

        # scalar "fitness" – odvodené číslo, používané len na logy/históriu
        self.fitness: Optional[float] = None

        # multi-kritériá:
        self.total_progress: float = 0.0
        self.time: float = float("inf")
        self.term: int = -999              # -999 = nevyhodnotený
        self.distance: float = 0.0         # celková prejdená vzdialenosť

    # ----------------- prístup k genómu -----------------

    @property
    def genome(self) -> np.ndarray:
        return self.policy.genome

    @genome.setter
    def genome(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32)
        if value.shape[0] != self.policy.genome_size:
            raise ValueError(
                f"Nový genóm má dĺžku {value.shape[0]}, "
                f"ale očakávané {self.policy.genome_size}"
            )
        self.policy.genome = value
        # po zmene váh je jedinec opäť "nevyhodnotený"
        self.fitness = None
        self.total_progress = 0.0
        self.time = float("inf")
        self.term = -999
        self.distance = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.policy.act(obs)

    # ----------------- multi-kritériové porovnávanie -----------------

    def ranking_key(self) -> Tuple[int, float, int, float]:
        """
        Lexikografický kľúč pre porovnanie jedincov.

        Poradie:
          1. term            (väčšie je lepšie; 1 > 0 > -1 > -3)
          2. total_progress  (vyššie percento trate je lepšie)
          3. time_bucket     (nižší čas v 1s bucketoch je lepší)
          4. distance        (menšia prejdená vzdialenosť je lepšia)

        Keďže v EvolutionTrainer používame sort(reverse=True),
        "najlepší" jedinec bude mať NAJväčší kľúč.
        """
        term = int(self.term)
        progress = float(self.total_progress)
        dist = float(self.distance)
        t = float(self.time)

        # 1-second discrete buckets so distance can act as a meaningful tiebreaker.
        if np.isfinite(t):
            time_bucket = int(np.floor(t))
        else:
            time_bucket = 10**9

        # smaller time_bucket / distance => use negatives for reverse sorting
        return (term, progress, -time_bucket, -dist)

    def compute_scalar_fitness(self) -> float:
        """
        Vyrobí skalárnu "fitness" len na logovanie / históriu.

        Má rovnakú prioritu kritérií ako ranking_key:
          (term, progress, time_bucket, distance)
        Používa sa len na logovanie / grafy, nie na samotný výber jedincov.
        """
        term, progress, neg_time_bucket, neg_dist = self.ranking_key()
        time_bucket = -neg_time_bucket
        dist = -neg_dist

        # Odhadované rozsahy:
        #  - term ∈ {...,-3,-2,-1,0,1}
        #  - progress ∈ [0,100]
        #  - time_bucket ≈ [0, 300]
        #  - distance ≈ [0, pár tisíc]
        #
        # Váhy zvolíme tak, aby:
        #  - term dominoval všetkému
        #  - progress dominoval time_bucket/distance
        #  - time_bucket dominoval distance
        A = 1_000_000_000.0  # váha pre term
        B = 1_000_000.0      # váha pre progress
        C = 10_000.0         # váha pre time_bucket
        D = 1.0              # váha pre distance

        return term * A + progress * B - time_bucket * C - dist * D

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

    # ----------------- genetické operácie -----------------

    def copy(self) -> "Individual":
        """
        Hlboká kópia jedinca (genóm aj hodnotenie sa kopírujú).
        """
        new = Individual(
            obs_dim=self.policy.obs_dim,
            hidden_dim=self.policy.hidden_dim,
            act_dim=self.policy.act_dim,
            genome=self.genome.copy(),
            action_scale=self.policy.action_scale.copy(),
        )
        new.fitness = self.fitness
        new.total_progress = self.total_progress
        new.time = self.time
        new.term = self.term
        new.distance = self.distance
        return new

    def mutate(self, mutation_prob: float = 0.1, sigma: float = 0.1) -> None:
        """
        Jednoduchá mutácia: s pravdepodobnosťou mutation_prob
        pridáme k danej váhe gaussovský šum ~ N(0, sigma^2).
        """
        g = self.genome
        mask = np.random.rand(g.size) < mutation_prob
        g[mask] += np.random.randn(mask.sum()).astype(np.float32) * sigma

        # po mutácii je jedinec nevyhodnotený
        self.fitness = None
        self.total_progress = 0.0
        self.time = float("inf")
        self.term = -999
        self.distance = 0.0

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
            action_scale=self.policy.action_scale.copy(),
        )
