'''Support classes for fastener.


'''

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Tuple, Any, Set
from abc import ABC, abstractmethod
import sklearn.feature_selection
import numpy as np

import random_utils

Genes = List[bool]
FitnessFunction = Callable[["Genes"], Tuple["Result", Any]]
#for each item size it keeps lists of Items of this size
Population = Dict[int, List["EvalItem"]]
UnevaluatedPopulation = Dict[int, List["Item"]]


IntScaling = Callable[[int], int]
FloatScaling = Callable[[float], float]


def flatten_population(population: Population) -> List["EvalItem"]:
    '''Creates array form dictionary.

    Transforms dictionary with List values into a single List with
    values form the other lists.

    Args:
        population: Dictionary to be transformed.
    
    Returns: 
        A List with values from all the other lists.
    '''
    return list(itertools.chain(*population.values()))


@dataclass
class Item:
    '''Class representing individual.

    An object that represents an individual in genethic algorithm 
    (a set of features to be used) and can be evaluated.

    Attributes:
        genes: An array of booleans marking which features are are
            selected.
        size: An integer count of selected features
        number: An integer which written in binary represents the
            active features (in reversed order).
        generation: An integer representing the generation the
            individual was created in.
        parent_a: An array of booleans representing genes from a parent
            (if parent exists).
        parent_b: An array of booleans representing genes from a parent
            (if parent exists).

    '''

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
        '''Post init function.

        After genes, geenration and parents are set size and number
        are calculated
        '''
        self.size = sum(self.genes)
        self.number = Item.to_number(self.genes)

    def evaluate(self, fitness_function: FitnessFunction) -> "EvalItem":
        '''Evaluates item.

        Calculates the result of the feature set and returns an 
        EvalItem object (Item with result attribute).

        Args:
            fitness_function: Function used to calculate result.

        Returns: 
            An EvalItem object.
        '''
        result = fitness_function(self.genes)[0]
        return EvalItem(self.genes, self.generation, self.parent_a,
                        self.parent_b, result)

    def __eq__(self, other: Any) -> bool:
        '''Checks if this item object is equivalent to other.

        Checks if passed variable (other) is of type Item and if it is
        equivalent to the self object.

        Args:
            other: A variable for which we wish to check if it is
                equivalent to self.
        
        Returns:
            True if the passed variable is of type Item and is
            equivalent to self and False otherwise. 
        '''
        return isinstance(other, Item) and self.number == other.number

    def __hash__(self) -> int:
        '''Returns hash value of number attribute.
        '''
        return hash(self.number)

    @classmethod
    def num_to_genes(cls, num: int, num_genes: int) -> "Genes":
        '''Converts number to genes array it represents.

        Converts number to binary and then sets the feature values
        accordingly to the string of ones and zeros. At the end it adds
        zeros if needed (for the not selected features) 

        Args:
            num: The number we wish to convert to genes array.
            num_genes: Number of all genes (active and inactive).

        Returns:
            An array of booleans (genes), for the given number.            
        '''
        return list(
            map(lambda x: bool(int(x)),
                reversed(bin(num)[2:].zfill(num_genes))))
    
    @classmethod
    def to_number(cls, genes: Genes) -> int:
        '''Converts array of genes to number.

        Loops throug the array of booleans and builds a string of ones
        and zeros then converts it to a decimal.

        Args:
            genes: An array of booleans (genes) to be converted to
                number.
        
        Returns:
            An integer representing the selected features.
        '''
        return int(''.join(map(str, map(int, reversed(genes)))), 2)

    @staticmethod
    def gene_union(g1: Genes, g2: Genes) -> Genes:
        '''Calculates a union of g1 and g2.

        Creates a new array of booleans in which i-th gene is True
        if i-th gene in g1 is True or i-th gene in g2 is True. It is
        used as a mating strategy.

        Args:
            g1: An array of booleans (genes).
            g2: An array of booleans (genes).

        Returns:
            An array of booleans (genes) in which i-th gene is True
            if i-th gene in g1 is True or i-th gene in g2 is True.
        '''
        return [i or j for i, j in zip(g1, g2)]

    @staticmethod
    def gene_intersection(g1: Genes, g2: Genes) -> Genes:
        '''Calculates an intersection of g1 and g2.

        Creates a new array of booleans in which i-th gene is True
        if i-th gene in g1 is True and i-th gene in g2 is True. It is
        used as a mating strategy.

        Args:
            g1: An array of booleans (genes).
            g2: An array of booleans (genes).

        Returns:
            An array of booleans (genes) in which i-th gene is True
            if i-th gene in g1 is True and i-th gene in g2 is True.
        '''
        return [i and j for i, j in zip(g1, g2)]

    @classmethod
    def from_genes(cls, genes: Genes, generation: int = 0) -> "Item":
        '''Creates Item from genes.

        Creates Item object with specified gened and generation.

        Args:
            genes: An boolean array of "genes".
            generation: An integer representing the generation we want
                the created item to be.

        Returns:
            An Item object we created from genes.
        '''
        return Item(genes, generation, None, None)

    @classmethod
    def indexed_symmetric_difference(cls, g1: Genes, g2: Genes) -> List[int]:
        '''Calculates indexes where genes differ.

        Loops through a pair of genes and stores indexes of places
        where the two differ in an array.

        Args:
            g1: An array of booleans (genes).
            g2: An array of booleans (genes).
        
        Returns:
            An integer array of indexes where the boolean array differ
        '''
        # XOR
        return [ind for ind, (i, j) in enumerate(zip(g1, g2)) if (i != j)]


@dataclass
class EvalItem(Item):
    '''Evaluated item.

    A child class of Item object (has all of the attributes + result).

    Attributes:
        result: A Result object containing a score of the evaluatin
            function for this set of attributes.

    '''

    result: "Result"

    # Same sized items are compared according to their result
    def __lt__(self, other: "EvalItem") -> bool:
        '''Check if other EvalItem object has larger score.

        Checks if self object has the same number of attributes
        (as other) and the score of self object is lower.

        Args:
            other: An EvalItem object to be compared.
        
        Returns:
            True if self and other object have the same size and self
            result is lower than others.

        Raises:
            AssertionError: If objects have different size attributes
        '''
        assert self.size == other.size
        return self.size >= other.size and self.result < other.result

    def __eq__(self, other: Any) -> bool:
        '''Overrides __eq__ methode from Item class.

        Checks if passed variable (other) is of type EvalItem and if it is
        equivalent to the self object.

        Args:
            other: A variable for which we wish to check if it is
                equivalent to self.
        
        Returns:
            True if the passed variable is of type EvalItem and is
            equivalent to self and False otherwise. 
        '''
        return isinstance(other, EvalItem) and self.number == other.number

    def __hash__(self) -> int:
        '''Returns hash value of number attribute.
        '''
        return hash(self.number)

    def evaluate(self, fitness_function: FitnessFunction) -> "EvalItem":
        '''Overrides Evaluate method from Item class.

        Calculates the result of the feature set and returns an 
        EvalItem object (Item with result attribute).

        Args:
            fitness_function: Function used to calculate result.

        Returns: 
            Object self.
        '''
        return self

    def pareto_better(self, other: "EvalItem") -> bool:
        """Compares EvalItem objects.

        Checks if self has less (or equal) genes, but better
        (or at least similar result).

        Attributes:
            other: EvalItem object to be compared.

        Returns:
            True if self objects has less (or equal) features but
            larger (or equal) score and False otherwise.
        """
        return self.size <= other.size and other.result <= self.result


@dataclass
class Result:
    '''Result of an evaluated Item.

    Attributes:
        score: A float that holds score of evaluated function for a
            given set of features.
    '''

    score: float

    def __lt__(self, other: "Result") -> bool:
        return self.score < other.score

    def __le__(self, other: "Result") -> bool:
        return self.score <= other.score


class MatingStrategy(ABC):
    '''Mating strategy.

    An abstract class that defines the base of mate function but not
    how the mating is implemented.
    '''

    @abstractmethod
    def mate_internal(self, item1: Item, item2: Item) -> Genes:
        pass

    def mate(self, item1: Item, item2: Item, generation: int) -> Item:
        '''Initailizes mating and creates a new Item.

        Initializes mating strategy (subclass specific) with parents
        and from the output genes creates a new Item.

        Attributes:
            item1: First parent (Item object) to be used for mating.
            item2: Second parent (Item object) to be used for mating.
            generatinon: Integer representing generation of the
                offspring (result of mating).
        
        Returns: 
            Item object that is the result of mating.
        '''
        genes = self.mate_internal(item1, item2)

        item = Item(genes, generation, item1.genes, item2.genes)

        return item

    def use_data_information(self, train_data: np.array,
                             train_target: np.array) -> None:
        pass


class UnionMating(MatingStrategy):
    '''Union mating strategy.

    A subclass of MatingStrategy which defines mating strategy as a
    union between genes (if at least one of parents has the gene, the
    decandant will too).
    '''
    def mate_internal(self, item1: Item, item2: Item) -> Genes:
        '''Implementation of union mating.

        Calls a static function of Item object which creates an array
        of genes with or logic and returns it.

        Args:
            item1: An Item object we wish to mate.
            item2: An Item object we wish to mate.
        
        Returns:
            An array of booleans (genes) that is the result of mating.
        '''
        return Item.gene_union(item1.genes, item2.genes)


class IntersectionMating(MatingStrategy):
    '''Intersection mating strategy.

    A subclass of MatingStrategy which defines mating strategy as an
    intersection between genes (only if both parents have the gene, the
    decandant will too).
    '''
    def mate_internal(self, item1: Item, item2: Item) -> Genes:
        '''Implementation of intersection mating.

        Calls a static function of Item object which creates an array
        of genes with and logic and returns it.

        Args:
            item1: An Item object we wish to mate.
            item2: An Item object we wish to mate.
        
        Returns:
            An array of booleans (genes) that is the result of mating.
        '''
        return Item.gene_intersection(item1.genes, item2.genes)


class IntersectionMatingWithInformationGain(IntersectionMating):
    '''Intersection mating strategy with information gain.

    A subclass of IntersectionMating TODO
    '''
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
    '''TODO
    '''
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
    '''
        TODO
    '''
    mating_pool: List["EvalItem"]
    carry_over: List["EvalItem"]


class MatingSelectionStrategy(ABC):
    '''Mating selection strategy.

    An abstract class that defines how the current population will be
    processed (in terms of mating) but not how the mating pool will be
    choosen or how pairs from that pool will be generated. 
    
    Attributes:
        mating_strategy: Mating strategy to be used in the algorithm.
        overwrite_carry_over: TODO
    '''
    def __init__(self, mating_strategy: MatingStrategy,
                 overwrite_carry_over: bool = True
                 ) -> None:
        self.mating_strategy = mating_strategy
        self.overwrite_carry_over = overwrite_carry_over

    def process_population(self, population: Population,
                           current_generation_number: int) -> \
            UnevaluatedPopulation:
        '''Procesess mating of population.

        Choses the mating pol and generates mated population then joins
        the items from the last generation (that will be in the new
        generation ass well) with the items that are the result of
        mating.

        Args:
            population: A dictionary that represents the Current
                population (of EvalItem objects).
            current_generation_number: An integer which represents the
                current generation of mating.

        Returns:
            A dictionary that represents the new population 
            (of Item objects).
        '''
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
    '''Everyone with everyone mating selection strategy.

    A subclass of MatingSelectionStrategy in which a mating pool of
    size pool_size is chosen and every possible pair from this pool is
    mated.

    Attributes:
        pool_size: the size of the pool from which mating pairs are
            generated.
    '''
    def __init__(self, mating_strategy: MatingStrategy = UnionMating(),
                 pool_size: int = 10):
        super().__init__(mating_strategy)
        self.pool_size = pool_size

    def mating_pool(self, population: Population) -> MatingPoolResult:
        '''Chooses mating pool.
        
        Transforms population into a single list and randomly chooses
        a mating pool of size pool_size.

        Args:
            population: A dictionary with List[Item] values and form
                those Items the mating pool is choosen.
        
        Returns:
            A MatingPoolResult object.
        '''
        flatten = flatten_population(population)
        pool_size = self.pool_size if self.pool_size is not None else len(
            flatten)
        return MatingPoolResult(random_utils.choices(flatten, size=self.pool_size),
                                flatten)

    def mate_pool(self, mating_pool: List["EvalItem"],
                  current_generation: int = 0) -> List[Item]:
        '''Chooses each possible pair from mating pool and mates them.

        Iterates through all possible pairs of items in mating pool and
        calls mate (from choosen mating strategy) on the pair.
        Generation of the offspring is increased by 1.

        Args:
            mating_pool: A List of EvalItem objects from which pairs
                for mating will be chosen.
            current_generation: An integer which represents the current
                generation of mating.
        
        Returns:
            A List of Item objects (the result of mating).
        '''
        rtr: List["Item"] = []
        for i in range(len(mating_pool)):
            for j in range(i + 1, len(mating_pool)):
                rtr.append(
                    self.mating_strategy.mate(mating_pool[i], mating_pool[j],
                                              current_generation + 1))
        return rtr


class NoMating(MatingSelectionStrategy):
    '''No mating selection strategy.

    A subclass of MatingSelectionStrategy which defines a strategy
    where no mating is performed.
    '''

    def mating_pool(self, population: Population) -> MatingPoolResult:
        '''Creates a MatingPoolResult object.

        Accepts a population (a dictionary) and makes that population
        a mating pool.

        Args:
            population: A dictionary to be made a mating pool.

        Returns:
            A MatingPoolResult object with population as mating pool.

        Raises:
             AssertionError: TODO: zakaj morata bit enake dolžine?
        '''
        f_pop = flatten_population(population)
        assert len(f_pop) == len(population)
        return MatingPoolResult(
            f_pop, []
        )

    def mate_pool(self, mating_pool: List["EvalItem"],
                  current_generation: int = 0) -> List[Item]:
        '''Raises the generation number of items without mating them.

        Iterates through EvalItem objects in the list and creates new
        Item objects with the generation number increased by one and
        appends them to an array.

        Argy:
            mating_pool: A list of EvalItem objects.
            current_generation: An integer which represents the current
                generation of mating.
        
        Returns:
            A List of Item objects (equivalent to the input list but
            with increased generation).
        '''
        return [Item(item.genes, item.generation + 1, None, None, ) for item in
                mating_pool]

    def use_data_information(self, train_data: np.array,
                             train_target: np.array) -> None:
        '''Overrides method from superclass.
        '''
        pass


class MutationStrategy(ABC):
    '''Mutation strategy

    An abstract class that defines how the population will be processed
    (mutated) and how mutation of each item will be performed but not
    the actual mutation implementation.
    '''
    @abstractmethod
    def mutate_internal(self, item: "Item") -> Genes:
        pass

    def process_population(self, population: UnevaluatedPopulation) -> \
            UnevaluatedPopulation:
        '''Iterates over population and performs mutations.

        Iterates over dictionary entries and for each item
        (list of Item objects) again iterates through Item objects in
        it, performs mutation and appends it to the correct List in
        the dictionary (or creates a new one if it does not exist).

        Args:
            population: A dictionary with integers (Item.size) as keys and lists
                of items of that size.

        Returns:
            A dictionary with the same content as the input dictionary
            but with mutated Items.
        '''
        pop2: UnevaluatedPopulation = {}
        for num, items in population.items():
            for item in items:
                mutated = self.mutate(item)
                pop2.setdefault(mutated.size, []).append(mutated)

        return pop2

    def mutate(self, item: "Item") -> Item:
        '''Mutates a single item.

        Performs mutation on an item (actual mutation is to be
        implemented) and from creates a new Item object form the
        mutates genes.

        Args:
            item: The item to be mutated.

        Returns:
            An Item object with mutated genes.
        '''
        new_genes = self.mutate_internal(item)

        return Item(new_genes, item.generation, item.parent_a, item.parent_b)


class RandomFlipMutationStrategy(MutationStrategy):
    '''Random flip mutation strategy.

    A subclass of MutationStrategy class that defines mutation as a
    flip of a gene with some probability.

    Attributes:
        prob: A float representing the probability of a gene beeing
            flipped
    '''
    def __init__(self, prob: float = 1 / 100):
        self.prob = prob

    def mutate_internal(self, item: "Item") -> Genes:
        '''Mutation implementation.

        Loops over given genes and flips each one with probability
        prob.

        Args:
            item: An item object whose genes we want to mutate.

        Returns:
            An array of booleans (mutated genes).
        '''
        genes = [
            not j if random_utils.random() < self.prob else j
            for j in item.genes
        ]
        return genes
