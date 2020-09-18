import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Tuple, Any, Set
from abc import ABC, abstractmethod
import sklearn.feature_selection
import numpy as np

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
        self.size = sum(self.genes)
        self.number = Item.to_number(self.genes)

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
        return int(''.join(map(str, map(int, reversed(genes)))), 2)

    @staticmethod
    def gene_union(g1: Genes, g2: Genes) -> Genes:
        return [i or j for i, j in zip(g1, g2)]

    @staticmethod
    def gene_intersection(g1: Genes, g2: Genes) -> Genes:
        return [i and j for i, j in zip(g1, g2)]

    @classmethod
    def from_genes(cls, genes: Genes, generation: int = 0) -> "Item":
        return Item(genes, generation, None, None)

    @classmethod
    def indexed_symmetric_difference(cls, g1: Genes, g2: Genes) -> List[int]:
        # XOR
        return [ind for ind, (i, j) in enumerate(zip(g1, g2)) if (i != j)]


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
        self.scikit_information_gain = sklearn.feature_selection.\
            mutual_info_classif(train_data, train_target)

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
                 scaling=None) -> None:
        super().__init__(number)
        self.number = number or self.default_number
        self.scaling = scaling or self.default_scaling

    @staticmethod
    def default_scaling(x):
        return np.log1p(np.log1p(np.log1p(x)))

    def mate_internal(self, item1: Item, item2: Item) -> Genes:
        genes = super().mate_internal(item1, item2)
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
                 pool_size: int = 10):
        super().__init__(mating_strategy)
        self.pool_size = pool_size

    def mating_pool(self, population: Population) -> MatingPoolResult:
        flatten = flatten_population(population)
        pool_size = self.pool_size if self.pool_size is not None else len(
            flatten)
        return MatingPoolResult(random_utils.choices(flatten, size=self.pool_size),
                                flatten)

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
        genes = [
            not j if random_utils.random() < self.prob else j
            for j in item.genes
        ]
        return genes
