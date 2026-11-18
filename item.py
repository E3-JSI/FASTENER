import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Tuple, Any, Set
from abc import ABC, abstractmethod
import numpy as np

import fast_mi
import random_utils

Genes = List[bool]
FitnessFunction = Callable[["Genes"], Tuple["Result", Any]]
Population = Dict[int, List["EvalItem"]]
UnevaluatedPopulation = Dict[int, List["Item"]]


IntScaling = Callable[[int], int]
FloatScaling = Callable[[float], float]


def flatten_population(population: Population) -> List["EvalItem"]:
    return list(itertools.chain(*population.values()))


@dataclass
class Item:
    genes: Genes
    size: int = field(init=False)
    # Number representing genes
    # Number is in little endian encoding, so that increasing the number of
    # genes (padding at the end) does not change number (appending new features
    # to existing ones keeps current caching valid)
    number: int = field(init=False)
    generation: int

    # parent_a: Optional["Item"]
    # parent_b: Optional["Item"]
    parent_a: Optional[Genes]
    parent_b: Optional[Genes]

    def __post_init__(self) -> None:
        # One numpy view for both, instead of two Python passes over all p
        # genes: this runs for every candidate, and on wide data (p in the
        # thousands) it dominated the search's run time.
        arr = np.asarray(self.genes, dtype=bool)
        self.size = int(arr.sum())
        self.number = Item._array_to_number(arr)

    def evaluate(self, fitness_function: FitnessFunction) -> "EvalItem":
        result = fitness_function(self.genes)[0]
        return EvalItem(self.genes, self.generation, self.parent_a,
                        self.parent_b, result)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Item) and self.number == other.number

    def __hash__(self) -> int:
        return hash(self.number)

    @classmethod
    def num_to_genes(cls, num: int, num_genes: int) -> "Genes":
        return list(
            map(lambda x: bool(int(x)),
                reversed(bin(num)[2:].zfill(num_genes))))
    @classmethod
    def to_number(cls, genes: Genes) -> int:
        return cls._array_to_number(np.asarray(genes, dtype=bool))

    @staticmethod
    def _array_to_number(arr: np.ndarray) -> int:
        # Gene i is bit i (little endian), as the original's string encoding:
        # the same integer, computed from packed bytes instead of a string of
        # p characters.
        if arr.size == 0:
            return 0
        return int.from_bytes(np.packbits(arr, bitorder="little").tobytes(),
                              "little")

    @staticmethod
    def gene_union(g1: Genes, g2: Genes) -> Genes:
        return (np.asarray(g1, dtype=bool) | np.asarray(g2, dtype=bool)).tolist()

    @staticmethod
    def gene_intersection(g1: Genes, g2: Genes) -> Genes:
        return (np.asarray(g1, dtype=bool) & np.asarray(g2, dtype=bool)).tolist()

    @classmethod
    def from_genes(cls, genes: Genes, generation: int = 0) -> "Item":
        return Item(genes, generation, None, None)

    @classmethod
    def indexed_symmetric_difference(cls, g1: Genes, g2: Genes) -> List[int]:
        # XOR, as ascending indices
        return np.flatnonzero(np.asarray(g1, dtype=bool)
                              != np.asarray(g2, dtype=bool)).tolist()


@dataclass
class EvalItem(Item):
    result: "Result"

    # Same sized items are compared according to their result
    def __lt__(self, other: "EvalItem") -> bool:
        assert self.size == other.size
        return self.size >= other.size and self.result < other.result

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, EvalItem) and self.number == other.number

    def __hash__(self) -> int:
        return hash(self.number)

    def evaluate(self, fitness_function: FitnessFunction) -> "EvalItem":
        return self

    def pareto_better(self, other: "EvalItem") -> bool:
        """
        I have less genes, but better (or at least similar result)
        """
        return self.size <= other.size and other.result <= self.result


@dataclass
class Result:
    score: float

    def __lt__(self, other: "Result") -> bool:
        return self.score < other.score

    def __le__(self, other: "Result") -> bool:
        return self.score <= other.score


class MatingStrategy(ABC):
    @abstractmethod
    def mate_internal(self, item1: Item, item2: Item) -> Genes:
        pass

    def mate(self, item1: Item, item2: Item, generation: int) -> Item:
        genes = self.mate_internal(item1, item2)

        item = Item(genes, generation, item1.genes, item2.genes)

        return item

    def use_data_information(self, train_data: np.array,
                             train_target: np.array) -> None:
        pass


class UnionMating(MatingStrategy):

    def mate_internal(self, item1: Item, item2: Item) -> Genes:
        return Item.gene_union(item1.genes, item2.genes)


class IntersectionMating(MatingStrategy):
    def mate_internal(self, item1: Item, item2: Item) -> Genes:
        return Item.gene_intersection(item1.genes, item2.genes)


class IntersectionMatingWithInformationGain(IntersectionMating):
    def __init__(self, number: Optional[IntScaling] = None) -> None:
        self.number = number or self.default_number
        self.scikit_information_gain: List[float] = []

    @staticmethod
    def default_number(x: int) -> int:
        return min(int(x / 2) + 1, x)

    def use_data_information(self, train_data: np.array,
                             train_target: np.array) -> None:
        # scikit-learn's estimator, computed by sorting each feature: the same
        # values and the same random draws, several times faster (fast_mi.py).
        self.scikit_information_gain = fast_mi.mutual_info_classif(
            train_data, train_target)

    def mate_internal(self, item1: Item, item2: Item) -> Genes:
        genes = super().mate_internal(item1, item2)
        ma = max(item1.size, item2.size)
        mi = min(item1.size, item2.size)

        sy_difference = Item. \
            indexed_symmetric_difference(item1.genes, item2.genes)

        sy_difference.sort(key=lambda x: self.scikit_information_gain[x],
                           reverse=True)
        for j in range(self.number(ma - mi)):
            genes[sy_difference[j]] = True
        return genes


class IntersectionMatingWithWeightedRandomInformationGain(
    IntersectionMatingWithInformationGain):
    def __init__(self, number: Optional[IntScaling] = None,
                 scaling=None, include_top_genes: bool = True) -> None:
        super().__init__(number)
        self.number = number or self.default_number
        self.scaling = scaling or self.default_scaling
        # True (the default, as in the original) starts from the parent
        # class's child -- the intersection plus the top-k genes by information
        # gain -- and then samples k more, so the top-k are always included and
        # a child gets up to 2k genes back (1.3k on average). False starts from
        # the plain intersection and adds only the weighted sample.
        self.include_top_genes = include_top_genes

    @staticmethod
    def default_scaling(x):
        return np.log1p(np.log1p(np.log1p(x)))

    def mate_internal(self, item1: Item, item2: Item) -> Genes:
        if self.include_top_genes:
            genes = super().mate_internal(item1, item2)
        else:
            genes = IntersectionMating.mate_internal(self, item1, item2)
        ma = max(item1.size, item2.size)
        mi = min(item1.size, item2.size)

        sy_difference = Item. \
            indexed_symmetric_difference(item1.genes, item2.genes)

        weights = \
            self.scaling([self.scikit_information_gain[i] for i in sy_difference])

        if sy_difference:  # If there are any different genes
            # They might all have 0 information gain
            if weights.sum() == 0:
                weights[:] = np.ones_like(weights) / len(weights)
            for ind in random_utils.choices(sy_difference, p=weights / weights.sum(),
                                            size=self.number(ma - mi)):
                genes[ind] = True

        return genes


@dataclass
class MatingPoolResult:
    mating_pool: List["EvalItem"]
    carry_over: List["EvalItem"]


class MatingSelectionStrategy(ABC):
    def __init__(self, mating_strategy: MatingStrategy,
                 overwrite_carry_over: bool = True
                 ) -> None:
        self.mating_strategy = mating_strategy
        self.overwrite_carry_over = overwrite_carry_over

    def process_population(self, population: Population,
                           current_generation_number: int) -> \
            UnevaluatedPopulation:
        result = self.mating_pool(population)
        mated = self.mate_pool(result.mating_pool, current_generation_number)

        new_population: UnevaluatedPopulation = {}

        for last in result.carry_over:
            new_population.setdefault(last.size, []).append(last)
        for new in mated:
            new_population.setdefault(new.size, []).append(new)

        return new_population

    @abstractmethod
    def mating_pool(self, population: Population) -> MatingPoolResult:
        pass

    @abstractmethod
    def mate_pool(self, mating_pool: List["EvalItem"],
                  current_generation: int = 0) -> List[Item]:
        pass

    def use_data_information(self, train_data: np.array,
                             train_target: np.array) -> None:
        self.mating_strategy.use_data_information(train_data, train_target)


class RandomEveryoneWithEveryone(MatingSelectionStrategy):

    def __init__(self, mating_strategy: MatingStrategy = UnionMating(),
                 pool_size: Optional[int] = 10,
                 allow_self_mating: bool = False):
        super().__init__(mating_strategy)
        self.pool_size = pool_size
        # The original draws the pool with replacement, so an item can land in
        # it twice and be mated with itself; its child is a copy of the parent,
        # which costs a crossover slot. Measured at ~13% of pairs with
        # pool_size=3. True restores that draw exactly, random stream included,
        # for reproducing results produced before this was changed.
        self.allow_self_mating = allow_self_mating

    def mating_pool(self, population: Population) -> MatingPoolResult:
        flatten = flatten_population(population)
        pool_size = self.pool_size if self.pool_size is not None else len(
            flatten)
        if self.allow_self_mating:
            pool = random_utils.choices(flatten, size=pool_size)
        else:
            # Without replacement the pool can be no larger than the population.
            indices = random_utils.choices(
                len(flatten), size=min(pool_size, len(flatten)), replace=False)
            pool = [flatten[i] for i in indices]
        return MatingPoolResult(pool, flatten)

    def mate_pool(self, mating_pool: List["EvalItem"],
                  current_generation: int = 0) -> List[Item]:
        rtr: List["Item"] = []
        for i in range(len(mating_pool)):
            for j in range(i + 1, len(mating_pool)):
                rtr.append(
                    self.mating_strategy.mate(mating_pool[i], mating_pool[j],
                                              current_generation + 1))
        return rtr


class NoMating(MatingSelectionStrategy):
    def mating_pool(self, population: Population) -> MatingPoolResult:
        f_pop = flatten_population(population)
        assert len(f_pop) == len(population)
        return MatingPoolResult(
            f_pop, []
        )

    def mate_pool(self, mating_pool: List["EvalItem"],
                  current_generation: int = 0) -> List[Item]:
        return [Item(item.genes, item.generation + 1, None, None, ) for item in
                mating_pool]

    def use_data_information(self, train_data: np.array,
                             train_target: np.array) -> None:
        pass


class MutationStrategy(ABC):
    @abstractmethod
    def mutate_internal(self, item: "Item") -> Genes:
        pass

    def process_population(self, population: UnevaluatedPopulation) -> \
            UnevaluatedPopulation:
        pop2: UnevaluatedPopulation = {}
        for num, items in population.items():
            for item in items:
                mutated = self.mutate(item)
                pop2.setdefault(mutated.size, []).append(mutated)

        return pop2

    def mutate(self, item: "Item") -> Item:
        new_genes = self.mutate_internal(item)

        return Item(new_genes, item.generation, item.parent_a, item.parent_b)


class RandomFlipMutationStrategy(MutationStrategy):
    def __init__(self, prob: float = 1 / 100):
        self.prob = prob

    def mutate_internal(self, item: "Item") -> Genes:
        # One draw per gene, as before, but in a single call: numpy's legacy
        # generator yields the same numbers for random(size=n) as for n calls
        # of random(), so the random stream -- and every later draw -- is
        # unchanged.
        flip = random_utils.random(len(item.genes)) < self.prob
        genes = (np.asarray(item.genes, dtype=bool) ^ flip).tolist()
        return genes


class SwapStrategy(ABC):
    """Size-preserving replacement operator for the pruning step.

    FASTENER's pruning only ever tests *deleting* the weakest feature of a
    subset. No operator in the algorithm preserves the subset size: mating pulls
    a child towards the middle size, pruning strictly decreases it, and mutation
    holds the size constant only when two bits happen to flip at once. A swap is
    that missing operator: it drops one selected feature and inserts one
    unselected feature in its place, so the candidate has the same size as its
    parent and competes directly against the front entry at that size.

    The strategy only *proposes* candidates; the optimizer evaluates them and
    passes them through the usual Pareto update. Swapping is opt-in -- with no
    swap strategy the optimizer runs the original deletion-only pruning.
    """

    def __init__(self, number_of_swaps: int = 1) -> None:
        # How many replacement candidates to propose per pruned subset. Each one
        # costs a model fit, so the default is a single candidate.
        self.number_of_swaps = number_of_swaps

    def use_data_information(self, train_data: np.array, train_target: np.array,
                             information_gain: Optional[np.array] = None) -> None:
        """Hook mirroring ``MatingStrategy.use_data_information``.

        ``information_gain`` is the mutual-information vector the mating
        strategy has already computed over the whole dataset, handed over so a
        guided swap does not pay for a second ``mutual_info_classif`` pass. It
        is ``None`` when the mating strategy does not compute one.
        """

    @abstractmethod
    def select_removal(self, genes: Genes,
                       changes: List[Tuple[float, int]]) -> int:
        """Index of the selected feature to drop.

        ``changes`` are the permutation-importance results the pruning step has
        already computed, as ``(score_drop, gene_index)`` sorted ascending, so
        ``changes[0]`` is the feature the model minds losing least.
        """

    @abstractmethod
    def select_insertions(self, genes: Genes, removed: int,
                          count: int) -> List[int]:
        """Indices of unselected features to try in the removed one's place."""

    def propose(self, item: "Item",
                changes: List[Tuple[float, int]]) -> List["Item"]:
        """Swap candidates for ``item``, unevaluated. Empty if none apply."""
        if not changes or all(item.genes):
            # Nothing to remove, or nothing left to insert.
            return []

        removed = self.select_removal(item.genes, changes)
        candidates: List["Item"] = []
        seen: Set[int] = set()
        for added in self.select_insertions(item.genes, removed,
                                            self.number_of_swaps):
            if added in seen or item.genes[added]:
                continue
            seen.add(added)
            genes = list(item.genes)
            genes[removed] = False
            genes[added] = True
            candidates.append(Item(genes, item.generation + 1, None, None))
        return candidates


class RandomSwapStrategy(SwapStrategy):
    """Unguided swap: random feature out, uniformly random feature in.

    The ablation arm. It uses neither the permutation scores nor the mutual
    information, so comparing it against the guided strategy isolates the value
    of the guidance rather than of swapping as such.
    """

    def select_removal(self, genes: Genes,
                       changes: List[Tuple[float, int]]) -> int:
        on_genes = np.where(genes)[0]
        return int(random_utils.choices(on_genes))

    def select_insertions(self, genes: Genes, removed: int,
                          count: int) -> List[int]:
        off_genes = np.where(np.logical_not(genes))[0]
        if not off_genes.size:
            return []
        return [int(i) for i in random_utils.choices(
            off_genes, size=min(count, off_genes.size), replace=False)]


class InformationGainSwapStrategy(SwapStrategy):
    """Guided swap: permutation-weakest feature out, MI-weighted feature in.

    Both ends reuse information FASTENER already computes -- the permutation
    scores from the pruning step itself, and the mutual-information vector from
    the mating strategy -- so the guidance costs no extra scoring pass. Sampling
    the inserted feature (rather than taking the MI-best one) keeps the search
    stochastic; on a dataset like MADELON the MI-best unselected feature would
    otherwise be proposed over and over, and `cached_fitness` would make every
    repeat a no-op.
    """

    def __init__(self, number_of_swaps: int = 1, scaling=None) -> None:
        super().__init__(number_of_swaps)
        self.scaling = scaling or \
            IntersectionMatingWithWeightedRandomInformationGain.default_scaling
        self.scikit_information_gain: List[float] = []

    def use_data_information(self, train_data: np.array, train_target: np.array,
                             information_gain: Optional[np.array] = None) -> None:
        if information_gain is not None and len(information_gain):
            # Reuse the mating strategy's vector: same numbers, no second pass.
            self.scikit_information_gain = information_gain
        else:
            self.scikit_information_gain = fast_mi.mutual_info_classif(
                train_data, train_target)

    def select_removal(self, genes: Genes,
                       changes: List[Tuple[float, int]]) -> int:
        # changes is sorted ascending by score drop, so the first entry is the
        # feature whose permutation cost the least -- the same feature the
        # original pruning step would have deleted.
        return int(changes[0][1])

    def select_insertions(self, genes: Genes, removed: int,
                          count: int) -> List[int]:
        off_genes = np.where(np.logical_not(genes))[0]
        if not off_genes.size:
            return []

        weights = np.asarray(
            self.scaling([self.scikit_information_gain[i] for i in off_genes]),
            dtype=float)
        weights[~np.isfinite(weights)] = 0.0
        weights[weights < 0] = 0.0

        size = min(count, off_genes.size)
        if weights.sum() == 0:
            # Every candidate is uninformative -- fall back to uniform.
            weights = np.ones_like(weights)
        elif int((weights > 0).sum()) < size:
            # Sampling without replacement needs at least `size` non-zero
            # weights; keep zero-information features reachable but negligible.
            weights = weights + 1e-12

        return [int(i) for i in random_utils.choices(
            off_genes, p=weights / weights.sum(), size=size, replace=False)]
