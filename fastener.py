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
    MatingStrategy, MutationStrategy, MatingSelectionStrategy, SwapStrategy, \
    RandomSwapStrategy, InformationGainSwapStrategy
import seeding
import guidance
from guidance import GuidanceContext, ProposalContext

Front = Dict[int, EvalItem]


class FitBudgetExhausted(Exception):
    """Raised internally when a run has spent its `max_model_fits` budget.

    Caught by `mainloop`, which stops the search. It exists so the budget is
    exact: comparing two variants of the algorithm is only meaningful if they
    are stopped at the same number of model fits, and checking the budget once
    per generation would let a generation overshoot it by however many subsets
    it happened to evaluate.
    """


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

    # Stop once this many models have been fitted, instead of running all
    # `number_of_rounds` generations. None (the default) means no cap, i.e. the
    # original round-limited behaviour.
    #
    # Fits are counted, not evaluations: with `cache_fitness_function` on, a
    # subset that has been seen before costs nothing and is not counted, so this
    # is a budget of UNIQUE subsets evaluated. Permutation scoring during
    # pruning is not counted either -- it only predicts, it does not fit.
    max_model_fits: Optional[int] = None

    # Write log/<output_folder>/generation_N.pickle every generation. The
    # pickle holds the whole population, front, mated and mutated sets AND the
    # full fitness cache, so on a wide data set it runs to several megabytes per
    # generation: measured at 166 MB and ~10 minutes for 40 generations on
    # ALLAML (7129 features), which made the logging, not the search, the
    # dominant cost of a run. Nothing in experiments/ reads these files, so a
    # batch of hundreds of runs turns them off; the default keeps the original
    # behaviour for single interactive runs.
    dump_generation_logs: bool = True

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
                 config: Optional[Config] = None,
                 swap_strategy: Optional[SwapStrategy] = None,
                 seed_filter: Optional[str] = None,
                 seed_size: int = 10,
                 boruta_max_iter: int = 100,
                 feature_prior: Optional["guidance.FeaturePrior"] = None,
                 prior_weight: float = 0.5,
                 prior_uses: Tuple[str, ...] = guidance.PRIOR_USES,
                 prior_initial_singles: int = 5,
                 guidance_context: Optional[GuidanceContext] = None,
                 subset_proposer: Optional["guidance.SubsetProposer"] = None,
                 proposer_every: int = 5) -> None:
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
        # Optional, off by default: with no swap strategy the pruning step is
        # exactly the original deletion-only one.
        self.swap_strategy = swap_strategy

        self.initial_population = initial_population
        self.initial_genes = initial_genes

        # Optional, off by default: start from a filter's top-k nested
        # subsets (see seeding.py), added to any initial genes given above.
        if seed_filter is not None and seed_filter not in seeding.SEED_FILTERS:
            raise ValueError(f"seed_filter must be one of "
                             f"{seeding.SEED_FILTERS} or None, "
                             f"not {seed_filter!r}")
        self.seed_filter = seed_filter
        self.seed_size = seed_size
        self.boruta_max_iter = boruta_max_iter
        # Filled in by prepare_loop: the filter's order (first `seed_size`
        # features) and the model fits it cost.
        self.seed_ranking: Optional[List[int]] = None
        self.seed_fits: int = 0

        # Optional, off by default: outside guidance (guidance.py). A feature
        # prior biases mating, mutation and the initial singles; a subset
        # proposer adds subsets to try every `proposer_every` generations.
        unknown = set(prior_uses) - set(guidance.PRIOR_USES)
        if unknown:
            raise ValueError(f"prior_uses must be among {guidance.PRIOR_USES}, "
                             f"not {sorted(unknown)}")
        if not 0.0 <= prior_weight <= 1.0:
            raise ValueError(f"prior_weight must be in [0, 1], not {prior_weight}")
        if proposer_every < 1:
            raise ValueError("proposer_every must be at least 1")
        self.feature_prior = feature_prior
        self.prior_weight = prior_weight
        self.prior_uses = tuple(prior_uses)
        self.prior_initial_singles = prior_initial_singles
        self.guidance_context = guidance_context or GuidanceContext(
            n_features=number_of_genes,
            feature_names=[f"f{i}" for i in range(number_of_genes)])
        if self.guidance_context.n_features != number_of_genes:
            raise ValueError("guidance_context.n_features differs from "
                             "number_of_genes")
        self.subset_proposer = subset_proposer
        self.proposer_every = proposer_every
        # Filled in by prepare_loop / the rounds: the resolved prior, the
        # combined weights, the singles drawn from them, and every proposer
        # call with what it returned and what it cost.
        self.prior_scores: Optional[np.ndarray] = None
        self.feature_weights: Optional[np.ndarray] = None
        self.prior_singles: List[List[int]] = []
        self.proposal_log: List[Dict[str, Any]] = []

        if reset_to_front_predicate is None:
            self.reset_to_front_predicate = EntropyOptimizer.default_reset_to_pareto
        else:
            self.reset_to_front_predicate = reset_to_front_predicate

        self.cache_counter: CounterType[int] = Counter()
        self.cache_data: Dict[int, Tuple[Result, Any]] = {}

        # Number of models actually fitted, and why the run ended. Both are
        # reported by the experiment runner.
        self.model_fits: int = 0
        self.stop_reason: Optional[str] = None
        self.rounds_completed: int = 0

        # What the pruning step did, cumulative over the run. Without these a
        # swap arm that matches stock FASTENER cannot say whether its swaps
        # lost every comparison or were barely ever tried. "kept" counts a
        # candidate still holding its size slot after the front update.
        self.prune_stats: CounterType[str] = Counter({
            "pruned_items": 0,
            "deletion_fits": 0, "deletion_kept": 0,
            "swap_proposed": 0, "swap_fits": 0, "swap_kept": 0,
        })
        # Numbers of the subsets a swap put on the front, so the front at any
        # point can say how many of its entries a swap made.
        self.swap_made: Set[int] = set()

        self.population: Population = {}
        self.pareto_front: Front = {}

        self.fitness_function = self._fitness_function

    def train_model(self, genes: "Genes") -> Any:
        return self._model().fit(self.train_data[:, genes], self.train_target)

    def _fitness_function(self, genes: "Genes") -> "Tuple[Result, Any]":
        # The single place a model is fitted, so the single place to count.
        # Everything that costs a fit passes through here -- mated and mutated
        # candidates, the pruning step's base evaluation, its deletion candidate
        # and any swap candidate -- so nothing can spend budget unmetered.
        if self.config.max_model_fits is not None and \
                self.model_fits >= self.config.max_model_fits:
            raise FitBudgetExhausted(
                f"budget of {self.config.max_model_fits} model fits is spent")
        self.model_fits += 1
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
            candidates = self.purge_item_candidates(item)
            # The deletion candidate is always first and always one smaller;
            # any further candidate is a same-sized swap, which competes
            # against the front entry at `num` instead.
            assert candidates[0].size == num - 1
            new_items.append((candidates[0], False))
            new_items.extend((swapped, True) for swapped in candidates[1:])

        better = 0
        new = 0
        for item, _ in new_items:
            if item.size in self.pareto_front:
                if item.result > self.pareto_front[item.size].result:
                    self.pareto_front[item.size] = item
                    better += 1
            else:
                self.pareto_front[item.size] = item
                new += 1

        # print("Purged:", "better:", better, "new:", new)

        self.remove_pareto_non_optimal()

        for item, is_swap in new_items:
            if self.pareto_front.get(item.size) is not item:
                continue
            if is_swap:
                self.prune_stats["swap_kept"] += 1
                self.swap_made.add(item.number)
            else:
                self.prune_stats["deletion_kept"] += 1

    def front_from_swap(self) -> int:
        """How many entries of the current front a swap put there."""
        return sum(item.number in self.swap_made
                   for item in self.pareto_front.values())

    def purge_item_with_information_gain(self, item: "Item") -> "EvalItem":
        """The deletion candidate for `item`, as in the original algorithm."""
        return self.purge_item_candidates(item)[0]

    def purge_item_candidates(self, item: "Item") -> "List[EvalItem]":
        """Pruning candidates for `item`, evaluated.

        The first is always the original deletion candidate: the subset without
        the feature whose permutation cost the score least. If a swap strategy
        is configured, same-sized replacement candidates follow -- they reuse
        the permutation scores computed here, so a swap costs one model fit and
        no extra scoring pass.
        """
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

        self.prune_stats["pruned_items"] += 1
        fits = self.model_fits
        candidates = [Item(genes2, item.generation + 1, None, None)
                      .evaluate(self.fitness_function)]
        self.prune_stats["deletion_fits"] += self.model_fits - fits

        if self.swap_strategy is not None:
            for swapped in self.swap_strategy.propose(item, changes):
                self.prune_stats["swap_proposed"] += 1
                fits = self.model_fits
                candidates.append(swapped.evaluate(self.fitness_function))
                # Zero when the subset was already in the cache.
                self.prune_stats["swap_fits"] += self.model_fits - fits

        return candidates

    def filter_seed(self) -> List[List[int]]:
        """The seed filter's top-k nested subsets, `seed_size` of them.

        Boruta's model fits are added to `model_fits`, so a fit-capped run
        pays for its seed out of the same budget. mRMR fits no model.
        """
        if self.seed_filter == "mrmr":
            order = seeding.rank_mrmr(self.train_data, self.train_target,
                                      self.seed_size,
                                      random_state=self.config.random_seed)
            fits = 0
        else:
            order, _, fits = seeding.rank_boruta(
                self._model, self.train_data, self.train_target,
                random_state=self.config.random_seed,
                max_iter=self.boruta_max_iter)
        self.seed_ranking = [int(i) for i in order[:self.seed_size]]
        self.seed_fits = fits
        self.model_fits += fits
        return seeding.nested_prefixes(order, self.seed_size)

    def mating_information_gain(self) -> Optional[np.array]:
        """The mating strategy's mutual-information vector, if it has one."""
        strategy = getattr(self.mating_selection_strategy, "mating_strategy",
                           None)
        vector = getattr(strategy, "scikit_information_gain", None)
        if vector is None or not len(vector):
            return None
        return vector

    def __getstate__(self):
        # The hooks may hold clients or closures that do not pickle; their
        # results (prior_scores, proposal_log) stay in the state.
        state = self.__dict__.copy()
        if callable(state.get("feature_prior")):
            state["feature_prior"] = None
        state["subset_proposer"] = None
        return state

    def apply_feature_prior(self) -> List[List[int]]:
        """Resolve the prior, combine it with mutual information and hand the
        weights to the operators in `prior_uses`. Returns the initial singles
        drawn from the weights ("initial"), else []."""
        self.mating_selection_strategy.use_data_information(self.train_data,
                                                            self.train_target)
        information_gain = self.mating_information_gain()
        if information_gain is None:
            if "mating" in self.prior_uses:
                raise ValueError("prior_uses has 'mating', but the mating "
                                 "strategy uses no information gain")
            import fast_mi
            information_gain = fast_mi.mutual_info_classif(self.train_data,
                                                           self.train_target)
        self.prior_scores = guidance.resolve_prior(self.feature_prior,
                                                   self.guidance_context)
        self.feature_weights = guidance.combine(information_gain,
                                                self.prior_scores,
                                                self.prior_weight)
        if "mating" in self.prior_uses:
            self.mating_selection_strategy.mating_strategy \
                .scikit_information_gain = self.feature_weights
        if "mutation" in self.prior_uses:
            if not hasattr(self.mutation_strategy, "use_feature_weights"):
                raise ValueError("prior_uses has 'mutation', but the mutation "
                                 "strategy takes no feature weights (use "
                                 "guidance.PriorGuidedMutationStrategy)")
            self.mutation_strategy.use_feature_weights(self.feature_weights)
        if "initial" in self.prior_uses:
            self.prior_singles = guidance.sample_singles(
                self.feature_weights, self.prior_initial_singles)
        return self.prior_singles

    def apply_proposals(self, round_n: int) -> None:
        """Ask the proposer for subsets, evaluate them and add them to the
        population and the front. Logged whether or not the budget runs out
        part-way."""
        ctx = ProposalContext(
            guidance=self.guidance_context, generation=round_n,
            front=guidance.front_summary(self.pareto_front),
            model_fits=self.model_fits,
            max_model_fits=self.config.max_model_fits,
            feature_weights=(None if self.feature_weights is None
                             else [float(x) for x in self.feature_weights]))
        proposed = self.subset_proposer(ctx)
        entry = {"generation": round_n,
                 "proposed": [[int(i) for i in s] for s in proposed or []],
                 "evaluated": [], "fits": 0}
        self.proposal_log.append(entry)
        fits = self.model_fits
        try:
            for genes in guidance.proposals_to_genes(proposed,
                                                     self.number_of_genes):
                itm = Item(genes, round_n, None, None) \
                    .evaluate(self.fitness_function)
                entry["evaluated"].append(
                    {"features": [int(i) for i in np.flatnonzero(genes)],
                     "score": float(itm.result.score)})
                self.population.setdefault(itm.size, []).append(itm)
        finally:
            entry["fits"] = self.model_fits - fits
        for items in self.population.values():
            items.sort(reverse=True)
        self.update_front_from_population()
        self.remove_pareto_non_optimal()

    @staticmethod
    def default_reset_to_pareto(round_num, _population, _front):
        return round_num % 3 == 0

    def prepare_loop(self) -> None:
        if self.config.cache_fitness_function:
            self.fitness_function = self.cached_fitness

        if self.initial_population:
            self.population = self.evaluate_unevaluated(self.initial_population)

        initial_genes = list(self.initial_genes or [])
        # With a feature prior, the data information is computed first, so
        # the initial singles can be drawn from the combined weights.
        guided = self.feature_prior is not None
        if guided:
            initial_genes += [genes for genes in self.apply_feature_prior()
                              if genes not in initial_genes]
        if self.seed_filter is not None:
            initial_genes += [genes for genes in self.filter_seed()
                              if genes not in initial_genes]

        if initial_genes:
            for in_gene in initial_genes:
                zeros = np.zeros(self.number_of_genes, dtype=bool)
                zeros[in_gene] = 1
                itm = Item(list(zeros), 0, None, None).evaluate(self.fitness_function)
                self.population.setdefault(itm.size, []).append(itm)

        assert self.population, "Some sort of initial population must be provided"

        if not guided:
            self.mating_selection_strategy.use_data_information(
                self.train_data, self.train_target)

        if self.swap_strategy is not None:
            # The mutual-information vector lives on the mating strategy; hand
            # it over rather than computing a second one.
            self.swap_strategy.use_data_information(
                self.train_data, self.train_target,
                self.mating_information_gain())

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
            try:
                self.run_round(round_n)
            except FitBudgetExhausted:
                # The generation was abandoned part-way, so the front may be
                # holding entries that the interrupted round had not yet pruned.
                self.remove_pareto_non_optimal()
                self.stop_reason = "fit_budget"
                print(f"Stopping: {self.model_fits} model fits spent")
                return
            self.rounds_completed = round_n

        self.stop_reason = "rounds"

    def run_round(self, round_n: int) -> None:
        """One generation: mate, mutate, evaluate, update the front, maybe prune."""
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

        if self.subset_proposer is not None and \
                round_n % self.proposer_every == 0:
            self.apply_proposals(round_n)


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
        if self.config.dump_generation_logs:
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
