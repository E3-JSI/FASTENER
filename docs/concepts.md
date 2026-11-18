# How FASTENER searches

This page describes what the code does, in the order it does it. Names in
`code` are the classes and methods in `fastener.py` and `item.py`.

## Subsets as genes

A candidate subset is a list of `number_of_genes` booleans, `genes[i]` being
true when feature `i` is selected. An `Item` wraps the genes. It also holds
their `size` (the number of selected features), a `number` (the genes read
as a little-endian bit string, which serves as the cache key), the
generation that made it, and the parents' genes. An `EvalItem` is an `Item`
with a `Result`. A `Result` holds one float, `score`, and higher is better.

## Population and buckets

The population is a dict keyed by subset size, `{size: [EvalItem, ...]}`.
Each list is a *bucket*. After every generation each bucket is sorted by
score and cut to its best `Config.max_bucket_size` items (default 3). So the
population never holds more than three subsets of any one size, and it keeps
subsets of many sizes side by side.

## The Pareto front

`optimizer.pareto_front` is a dict `{size: EvalItem}` holding the best subset
found for each size. It is kept free of dominated entries: a subset is
removed when some smaller (or equal-sized) subset scores at least as well.
Read from the smallest size up, the scores on the front therefore strictly
increase. A tie goes to the smaller subset.

The front is the result of a run. FASTENER does not pick one subset for you.
Choose by your own rule, for example the smallest subset within ε of the best
score, and judge your choice on data the search did not see (see
[pitfalls.md](pitfalls.md#the-front-is-optimistic)).

## Before the first generation (`prepare_loop`)

1. With `Config.cache_fitness_function` (the default), fitness goes through
   a cache keyed by `Item.number`, so a subset is fitted once per run.
2. The initial population is built from whichever of these you gave:
   `initial_population` (unevaluated `Item`s by size), `initial_genes`
   (lists of feature indices), the singles drawn from a feature prior
   (`prior_uses` containing `"initial"`), and a filter seed's nested top-k
   subsets (`seed_filter`). Each is fitted and scored. Duplicates among
   `initial_genes`, the prior singles and the seed are evaluated once.
3. The mating strategy's `use_data_information(X_train, y_train)` runs once.
   For the information-gain strategies it estimates the mutual information
   of every feature with the target (`fast_mi.mutual_info_classif`). A feature
   prior, if given, is combined into these weights here. A guided swap
   strategy receives the same vector.
4. The optimizer is pickled to `log/<output_folder>/experiment.pickle`.

`mainloop` then cuts the buckets and builds the first front.

## One generation (`run_round`)

1. **Select and mate.** `RandomEveryoneWithEveryone` draws `pool_size` items
   (without replacement) from the whole population and mates every pair:
   a pool of 3 gives 3 children, a pool of 5 gives 10. All current items are
   also carried over into the next step.
2. **Mutate.** Every child *and* every carried-over item is mutated.
   `RandomFlipMutationStrategy(p)` flips each gene with probability `p`;
   `1 / number_of_genes` flips one gene per item on average. An item that
   comes out unchanged costs nothing, because of the cache.
3. **Deduplicate and evaluate.** Duplicates within a size are dropped, and
   empty subsets are skipped. Every remaining subset is fitted (or read from
   the cache) and scored.
4. **Cut and update.** Buckets are cut to `max_bucket_size`, the best item of
   each size updates the front, and dominated front entries are removed.
5. **Proposals** (if a `subset_proposer` is set, every `proposer_every`
   generations): the proposer's subsets are evaluated and join the
   population and the front.
6. **Prune and reset** (every `Config.reset_to_pareto_rounds` generations,
   default 5): the front is pruned (below), and the population is replaced
   by the front, one item per size.
7. If `Config.dump_generation_logs`, the generation is pickled to
   `log/<output_folder>/generation_N.pickle`.

The run ends after `Config.number_of_rounds` generations, or as soon as
`Config.max_model_fits` models have been fitted, whichever comes first.
`optimizer.stop_reason` says which.

## How mating uses mutual information

Every mating strategy in `item.py` starts from the parents' **intersection**,
except `UnionMating`. The information-gain strategies then give back some of
the genes in which the parents differ (their symmetric difference). How many
they give back depends on `k = |size(parent1) − size(parent2)|`.

* `IntersectionMatingWithInformationGain` adds the `number(k)` differing genes
  with the highest mutual information. The default `number(k)` is
  `min(k // 2 + 1, k)`, which gives 0, 1, 2, 2, 3, 3, … for k = 0, 1, 2, …
* `IntersectionMatingWithWeightedRandomInformationGain` (the usual choice)
  also samples `number(k)` differing genes *with replacement*, with
  probability proportional to `scaling(MI)`, where the default scaling is
  `log1p(log1p(log1p(MI)))`. With `include_top_genes=True` (the default) it
  adds the top genes first, as the 2020 code did. With `False` it adds only
  the sample.

Two parents of equal size have `k = 0`, so their child is their
intersection. Mating therefore tends to produce subsets smaller than the
larger parent, while mutation and the union of differing genes push the
other way. The size balance comes from the population holding every size at
once.

## Pruning (`purge_front_with_information_gain`)

For every front entry with more than one feature:

1. The model on that subset is fitted (normally a cache hit) and scored.
2. Each selected feature is shuffled in turn in the evaluation data. The
   evaluator is called with `shuffle_indices=[j]`, and the drop in score is
   the feature's permutation importance. This step predicts but does not fit.
3. The feature with the smallest drop is removed, and the smaller subset is
   fitted and scored. It takes the front slot one size down if it beats what
   is there.
4. With a `swap_strategy`, same-sized candidates are also tried. Each drops
   one feature and adds one unselected feature, competing for the slot at the
   same size. They cost one fit each.

`optimizer.prune_stats` counts what was tried and what was kept.

## What counts as a model fit

Every fit goes through `EntropyOptimizer._fitness_function`. That covers
mated and mutated candidates, the initial population, pruning, swaps and
proposals, and Boruta's iterations when it seeds the search.
`optimizer.model_fits` is the total. Cache hits and permutation scoring are
not fits. With the cache on, `max_model_fits` is therefore a budget of
*distinct* subsets evaluated.

## Randomness

All draws go through `random_utils`, which is numpy's global legacy
generator. `Config(random_seed=s)` seeds it when the `Config` is created. A
run is then repeatable if nothing else draws from `np.random` between
creating the `Config` and the end of `mainloop`. Your evaluator's shuffles
should use `random_utils.shuffle` for the same reason. The mutual-information
estimate draws from the same generator.
