import math
import os
import pickle
import time
from dataclasses import dataclass, field
from functools import wraps

import random_utils
from random_utils import shuffle
from typing import Dict, List, Callable, Any, Tuple, Optional, \
    Counter as CounterType, Set
import numpy as np
from collections import Counter

from item import Item, Result, Population, flatten_population, FitnessFunction, \
    Genes, EvalItem, RandomFlipMutationStrategy, RandomEveryoneWithEveryone, \
    IntersectionMating, UnionMating, IntersectionMatingWithInformationGain, \
    IntersectionMatingWithWeightedRandomInformationGain, UnevaluatedPopulation, \
    MatingStrategy, MutationStrategy, MatingSelectionStrategy

Front = Dict[int, EvalItem]


@dataclass
class LogData:
    generation: int
    population: Population
    front: Front
    mated: UnevaluatedPopulation
    mutated: UnevaluatedPopulation

    cache_counter: CounterType[int]
    cache_data: Dict[int, Result]
    random_state: Any
    timestamp: float = field(init=False)

    def __post_init__(self):
        self.timestamp = time.time()

    @staticmethod
    def discard_model(cache_data: Dict[int, Tuple[Result, Any]]) -> Dict[int, Result]:
        return {num: item[0] for num, item in cache_data.items()}

    def dump_log(self, config: "Config") -> None:
        pickle.dump(
            self,
            open(os.path.join(
                config.output_folder,
                f"generation_{self.generation}.pickle"),
                "wb")
        )


@dataclass
class Config:
    output_folder: str
    random_seed: int
    number_of_rounds: int = 1000  # Number of round of genetic algorithm
    max_bucket_size: int = 3  # Max number of items in same size bucket
    reset_to_pareto_rounds: Optional[int] = 5  # Reset population to Pareto front every n-rounds (False is never)

    cache_fitness_function: bool = True

    def __post_init__(self) -> None:
        if not self.reset_to_pareto_rounds:
            self.reset_to_pareto_rounds = self.number_of_rounds
        self.output_folder = os.path.join("log", self.output_folder)

        random_utils.seed(self.random_seed)


class EntropyOptimizer:
    def __init__(self,
                 model: Any,
                 train_data: np.array,
                 train_target: np.array,
                 evaluator: Callable[
                     [Any, "Genes", Optional[List[int]]], "Result"],
                 number_of_genes: int,
                 mating_selection_strategy: MatingSelectionStrategy,
                 mutation_strategy: MutationStrategy,
                 reset_to_front_predicate: Optional[
                     Callable[[int, Population, Front], bool]] = None,
                 initial_population: Optional[UnevaluatedPopulation] = None,
                 initial_genes: Optional[List[List[int]]] = None,
                 config: Optional[Config] = None) -> None:
        self._model = model
        self.evaluator = evaluator

        self.train_data = train_data
        self.train_target = train_target

        self.number_of_genes = number_of_genes
        if config is None:
            config = Config("data", 2020)

        self.config = config
        os.makedirs(self.config.output_folder, exist_ok=False)

        self.mating_selection_strategy = mating_selection_strategy
        self.mutation_strategy = mutation_strategy

        self.initial_population = initial_population
        self.initial_genes = initial_genes

        if reset_to_front_predicate is None:
            self.reset_to_front_predicate = EntropyOptimizer.default_reset_to_pareto
        else:
            self.reset_to_front_predicate = reset_to_front_predicate

        self.cache_counter: CounterType[int] = Counter()
        self.cache_data: Dict[int, Tuple[Result, Any]] = {}

        self.population: Population = {}
        self.pareto_front: Front = {}

        self.fitness_function = self._fitness_function

    def train_model(self, genes: "Genes") -> Any:
        return self._model().fit(self.train_data[:, genes], self.train_target)

    def _fitness_function(self, genes: "Genes") -> "Tuple[Result, Any]":
        model = self.train_model(genes)
        return self.evaluator(model, genes, None), model

    def cached_fitness(self, genes: "Genes") -> "Tuple[Result, Any]":
        #print("Trying:", np.where(genes))
        number = Item.to_number(genes)
        result = self.cache_data.get(number, None)
        self.cache_counter[number] += 1
        if result is None:
            result = self._fitness_function(genes)
            self.cache_data[number] = result
        else:
            #print("Hitted", self.cache_counter)
            pass
        return result

    def purge_front_with_information_gain(self):
        new_items = []
        for num, item in self.pareto_front.items():
            if num == 1:  # Can't remove features
                continue
            new_item = self.purge_item_with_information_gain(item)
            assert new_item.size == num - 1
            new_items.append(new_item)

        better = 0
        new = 0
        for item in new_items:
            if item.size in self.pareto_front:
                if item.result > self.pareto_front[item.size].result:
                    self.pareto_front[item.size] = item
                    better += 1
            else:
                self.pareto_front[item.size] = item
                new += 1

        # print("Purged:", "better:", better, "new:", new)

        self.remove_pareto_non_optimal()

    def purge_item_with_information_gain(self, item: "Item") -> "EvalItem":
        base_result, model = self.fitness_function(item.genes)
        on_genes = np.where(item.genes)[0]
        changes = [(1.0, -1) for _ in on_genes]

        for change_ind, gene_ind in enumerate(on_genes):
            sh_result = self.evaluator(model, item.genes, [change_ind])
            changes[change_ind] = (base_result.score - sh_result.score, gene_ind)
        changes.sort()  # Sort them so that first cause the smallest change
        # print(base_result)
        # print("CHANGES:", changes)
        # And therefore seem a good fit for removal
        # Maybe some random selection?
        genes2 = item.genes.copy()
        # Unset the gene with the smallest change
        genes2[changes[0][1]] = False

        rt_item = Item(genes2, item.generation + 1, None, None).\
            evaluate(self.fitness_function)
        return rt_item

    @staticmethod
    def default_reset_to_pareto(round_num, _population, _front):
        return round_num % 3 == 0

    def prepare_loop(self) -> None:
        if self.config.cache_fitness_function:
            self.fitness_function = self.cached_fitness

        if self.initial_population:
            self.population = self.evaluate_unevaluated(self.initial_population)

        if self.initial_genes:
            for in_gene in self.initial_genes:
                zeros = np.zeros(self.number_of_genes, dtype=bool)
                zeros[in_gene] = 1
                itm = Item(list(zeros), 0, None, None).evaluate(self.fitness_function)
                self.population.setdefault(itm.size, []).append(itm)

        assert self.population, "Some sort of initial population must be provided"

        self.mating_selection_strategy.use_data_information(self.train_data,
                                                            self.train_target)

        pickle.dump(
            self, open(
                os.path.join(self.config.output_folder, "experiment.pickle"),
                "wb")
        )

    def mainloop(self) -> None:

        self.prepare_loop()

        self.purge_oversize_buckets()
        self.update_front_from_population()

        for round_n in range(1, self.config.number_of_rounds + 1):
            print(f"Round: {round_n}")
            mated = self.mating_selection_strategy.process_population(
                self.population, round_n
            )
            mutated = self.mutation_strategy.process_population(mated)
            mutated = self.clear_duplicates_after_mating(mutated)
            # print(mutated)
            # print(len(mutated))
            # Evaluate new population
            # Do not evaluate "empty" genes
            self.population = self.evaluate_unevaluated(mutated)

            self.purge_oversize_buckets()
            self.update_front_from_population()
            bef = len(self.pareto_front)
            self.remove_pareto_non_optimal()
            aft = len(self.pareto_front)

            # print(len(flatten_population(self.population)), bef, aft)


            if self.config.reset_to_pareto_rounds and \
                    round_n % self.config.reset_to_pareto_rounds == 0:

                self.purge_front_with_information_gain()

                self.reset_population_to_front()
                # print("RESETING POPULATION TO FRONT")

            log_data = LogData(round_n, self.population, self.pareto_front, mated,
                               mutated, self.cache_counter,
                               LogData.discard_model(self.cache_data),
                               random_utils.get_state())

            # Log during evaluation
            log_data.dump_log(self.config)

    def purge_oversize_buckets(self) -> None:
        new_population: Population = {}
        for num, items in self.population.items():
            new_population[num] = sorted(items, reverse=True)[:self.config.max_bucket_size]
        self.population = new_population

    def update_front_from_population(self) -> None:
        # Assume that population is sorted
        for num, items in self.population.items():
            if num in self.pareto_front:
                if items:
                    self.pareto_front[num] = \
                        max(self.pareto_front[num], items[0])
            else:
                if items:
                    self.pareto_front[num] = items[0]

    @classmethod
    def clear_duplicates_after_mating(cls, un_pop: "UnevaluatedPopulation") \
            -> "UnevaluatedPopulation":
        new_un_pop: UnevaluatedPopulation = {}
        for num, items in un_pop.items():
            new: List[Item] = []
            taken: Set[int] = set()
            for item in items:
                if item.number not in taken:
                    taken.add(item.number)
                    new.append(item)
            new_un_pop[num] = new

        return new_un_pop

    def remove_pareto_non_optimal(self) -> None:
        # Make a copy as to not change dict during looping,
        # sort them by the number of genes
        sorted_items = list(sorted(
            self.pareto_front.items(), key=lambda item: item[0]
        ))

        current_peak = sorted_items[0][1]
        for j in range(1, len(sorted_items)):
            n, item = sorted_items[j]
            if current_peak.pareto_better(item):
                del self.pareto_front[n]
            else:
                current_peak = item

    def reset_population_to_front(self) -> None:
        new_population: Population = {}
        # Keep Pareto front in itemized order
        for num, item in sorted(self.pareto_front.items(),
                                key=lambda itm: itm[0]):
            new_population[num] = [item]
        self.population = new_population

    def evaluate_unevaluated(self, un_pop: "UnevaluatedPopulation") \
            -> Population:
        return {
            num: [item.evaluate(self.fitness_function) for item in items
                  if item.size]
            for num, items in un_pop.items()
        }


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

general_model = DecisionTreeClassifier
from dummy_data import XX_train, XX_test, labels_train, labels_test


def eval_fun(model: Any, genes: "Genes",
             shuffle_indices: Optional[List[int]] = None) -> "Result":
    test_data = XX_test[:, genes]
    if shuffle_indices:
        test_data = test_data.copy()
        for j in shuffle_indices:
            shuffle(test_data[:, j])
    pred = model.predict(test_data)
    res = Result(f1_score(labels_test, pred, average='weighted'))
    return res


def main() -> None:

    number_of_genes = XX_train.shape[1]

    initial_genes = [
        [0]
    ]

    # Select mating strategies
    mating = RandomEveryoneWithEveryone(
        pool_size=3,
        mating_strategy=IntersectionMatingWithWeightedRandomInformationGain())

    # Random mutation
    mutation = RandomFlipMutationStrategy(1 / number_of_genes)

    entropy_optimizer = EntropyOptimizer(
        general_model, XX_train, labels_train, eval_fun,
        number_of_genes, mating, mutation, initial_genes=initial_genes,
        config=Config(output_folder="dummy_data_output", random_seed=2020,
                      reset_to_pareto_rounds=5)
    )

    entropy_optimizer.mainloop()


if __name__ == '__main__':
    main()
