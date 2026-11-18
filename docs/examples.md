# Examples

Every recipe here comes from a script in [examples/](examples/) that runs
as is. The snippets leave out imports and data loading. The scripts have
them.

Most scripts share two helpers from [examples/_common.py](examples/_common.py):
`classification_data()` (breast cancer, 569 × 30, split 50/25/25 into
train / validation / test) and `ValidationEvaluator(X_val, y_val, score)`, a
picklable evaluator for any `score(y_true, y_pred)`.

These runs use one seed on one small data set, so they show *how* to use an
option, not whether it helps. Compare options with several seeds, at an equal
`max_model_fits`, and on held-out data.

| Script | Shows |
|---|---|
| [01_quickstart.py](examples/01_quickstart.py) | A complete run, choosing a subset from the front, testing it |
| [02_models_and_metrics.py](examples/02_models_and_metrics.py) | Trees, forests, pipelines; balanced accuracy, ROC AUC, a size penalty |
| [03_mating_and_mutation.py](examples/03_mating_and_mutation.py) | Every mating strategy, pool sizes, mutation rates, the 2020 behaviour |
| [04_starting_points_and_config.py](examples/04_starting_points_and_config.py) | `initial_genes`, `initial_population`, `Config` knobs, the cache |
| [05_budget_outputs_and_logs.py](examples/05_budget_outputs_and_logs.py) | `max_model_fits`, what a run reports, reading the log pickles |
| [06_swap_operator.py](examples/06_swap_operator.py) | Guided and random swaps in the pruning step |
| [07_filter_seeds.py](examples/07_filter_seeds.py) | mRMR and Boruta seeds, and the filters on their own |
| [08_llm_guidance.py](examples/08_llm_guidance.py) | LLM guidance: an LLM feature prior and an LLM subset proposer through OpenRouter |
| [09_regression.py](examples/09_regression.py) | Regression: MI for a continuous target, R² and −RMSE, a Boruta seed |
| [10_custom_strategies.py](examples/10_custom_strategies.py) | Your own mutation, crossover, parent selection and swap |

---

## 1. Basics

### The smallest complete run

```python
def evaluator(model, genes, shuffle_indices=None):
    X_eval = X_val[:, genes]
    if shuffle_indices:
        X_eval = X_eval.copy()
        for j in shuffle_indices:
            random_utils.shuffle(X_eval[:, j])
    return Result(f1_score(y_val, model.predict(X_eval), average="weighted"))

n = X_train.shape[1]
optimizer = EntropyOptimizer(
    model=DecisionTreeClassifier,
    train_data=X_train, train_target=y_train,
    evaluator=evaluator, number_of_genes=n,
    mating_selection_strategy=RandomEveryoneWithEveryone(
        pool_size=3,
        mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
    mutation_strategy=RandomFlipMutationStrategy(1 / n),
    initial_genes=[[i] for i in range(n)],
    config=Config(output_folder="quickstart", random_seed=2020,
                  number_of_rounds=50, dump_generation_logs=False))
optimizer.mainloop()
```

### Reading the front

```python
for size, item in sorted(optimizer.pareto_front.items()):
    features = numpy.flatnonzero(item.genes)
    print(size, item.result.score, [names[i] for i in features])
```

```
  1 features  score 0.9089  ['worst perimeter']
  2 features  score 0.9577  ['worst perimeter', 'worst concave points']
  3 features  score 0.9718  ['radius error', 'worst area', 'worst concave points']
```

### Choosing one subset and testing it

The front's scores were used to choose the subsets, so they are optimistic.
Pick by a rule, refit on train + validation, and score the result once on
the test split:

```python
best = max(item.result.score for item in optimizer.pareto_front.values())
size, chosen = min((s, it) for s, it in optimizer.pareto_front.items()
                   if it.result.score >= best - 0.01)   # smallest within 0.01
cols = numpy.flatnonzero(chosen.genes)
model = DecisionTreeClassifier().fit(X_rest[:, cols], y_rest)
print(f1_score(y_test, model.predict(X_test[:, cols]), average="weighted"))
```

---

## 2. Models and scores — [02](examples/02_models_and_metrics.py)

The model is a factory called with no arguments. The evaluator is any
picklable callable returning `Result(score)`, higher is better.

```python
# Hyperparameters with functools.partial
model = partial(DecisionTreeClassifier, max_depth=5, class_weight="balanced")

# A forest (each fit is slower, so give the search fewer rounds or a budget)
model = partial(RandomForestClassifier, n_estimators=30, n_jobs=-1, random_state=0)

# A pipeline: a module-level function, not a lambda (it must pickle)
def scaled_logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))

# Balanced accuracy on the validation split
evaluator = ValidationEvaluator(X_val, y_val, balanced_accuracy_score)

# A probability-based score: write the evaluator out
def auc_evaluator(model, genes, shuffle_indices=None):
    X_eval = X_val[:, genes]
    if shuffle_indices:
        X_eval = X_eval.copy()
        for j in shuffle_indices:
            random_utils.shuffle(X_eval[:, j])
    return Result(roc_auc_score(y_val, model.predict_proba(X_eval)[:, 1]))

# A score that charges for size (rarely needed: the front already
# gives one subset per size)
def penalised(model, genes, shuffle_indices=None):
    return Result(balanced(model, genes, shuffle_indices).score - 0.002 * sum(genes))
```

---

## 3. Mating and mutation — [03](examples/03_mating_and_mutation.py)

```python
# The default since 2026-09-30, and the published algorithm's choice
RandomEveryoneWithEveryone(
    pool_size=3,
    mating_strategy=IntersectionMatingWithWeightedRandomInformationGain())

# Only the weighted sample, without the top genes by MI
IntersectionMatingWithWeightedRandomInformationGain(include_top_genes=False)

# Your own number of genes to give back, and your own weight scaling
def half_the_difference(k):
    return k // 2

def gentler_scaling(mi):
    return numpy.log1p(numpy.asarray(mi))

IntersectionMatingWithWeightedRandomInformationGain(
    number=half_the_difference, scaling=gentler_scaling)

# Deterministic top-k by MI; unguided intersection; union
IntersectionMatingWithInformationGain()
IntersectionMating()
UnionMating()

# A pool of 5: 10 children per generation instead of 3. None: the whole population
RandomEveryoneWithEveryone(pool_size=5, mating_strategy=...)
RandomEveryoneWithEveryone(pool_size=None, mating_strategy=...)

# Reproduce results from the 2020 code exactly
RandomEveryoneWithEveryone(
    pool_size=3, allow_self_mating=True,
    mating_strategy=IntersectionMatingWithWeightedRandomInformationGain(
        include_top_genes=True))

# Mutation: on average one flip per subset, or three
RandomFlipMutationStrategy(1 / n)
RandomFlipMutationStrategy(3 / n)
```

The script runs each of these to 400 fits and prints the front sizes and the
best score for each.

---

## 4. Starting points and `Config` — [04](examples/04_starting_points_and_config.py)

```python
initial_genes=[[0]]                              # one feature
initial_genes=[[i] for i in range(n)]            # every single feature (n fits)
initial_genes=known_subsets + [[i] for i in range(n)]   # your own subsets, plus singles

# An explicit population of unevaluated Items, keyed by size
population = {}
for genes in random_triples:
    item = Item.from_genes(genes)
    population.setdefault(item.size, []).append(item)
EntropyOptimizer(..., initial_population=population)
```

```python
Config(output_folder="run", random_seed=3,
       number_of_rounds=30,
       max_bucket_size=6,            # keep 6 subsets per size, not 3
       reset_to_pareto_rounds=10,    # prune every 10 rounds, not 5
       cache_fitness_function=True,  # the default; False refits repeats
       dump_generation_logs=False)
```

With the cache, the script's run spends 254 fits where the same run without
it spends 400. 122 lookups were repeats.

---

## 5. Budgets, outputs and logs — [05](examples/05_budget_outputs_and_logs.py)

```python
config = Config(output_folder="budget", random_seed=11,
                number_of_rounds=100_000,  # high, so that the budget stops the run
                max_model_fits=300)
...
optimizer.mainloop()
optimizer.stop_reason        # "fit_budget"
optimizer.model_fits         # 300
optimizer.rounds_completed   # 36
optimizer.prune_stats        # {'pruned_items': 14, 'deletion_fits': 4, 'deletion_kept': 2, ...}
```

```python
# One pickle per generation in log/<output_folder>/
with open(f"{config.output_folder}/generation_36.pickle", "rb") as f:
    log = pickle.load(f)                  # fastener.LogData
log.front, log.population, log.cache_data, log.random_state

with open(f"{config.output_folder}/experiment.pickle", "rb") as f:
    before = pickle.load(f)               # the EntropyOptimizer before the search
```

The script also traces the best score on the front generation by generation
from those files.

---

## 6. The swap operator — [06](examples/06_swap_operator.py)

```python
EntropyOptimizer(..., swap_strategy=None)                                   # published
EntropyOptimizer(..., swap_strategy=InformationGainSwapStrategy())          # guided
EntropyOptimizer(..., swap_strategy=InformationGainSwapStrategy(number_of_swaps=3))
EntropyOptimizer(..., swap_strategy=RandomSwapStrategy())                   # ablation

stats = optimizer.prune_stats
stats["swap_proposed"], stats["swap_fits"], stats["swap_kept"]
optimizer.front_from_swap()   # front entries a swap put there
```

`swap_fits` can be lower than `swap_proposed`: a proposed subset that is
already in the cache costs no fit.

---

## 7. Filter seeds — [07](examples/07_filter_seeds.py)

```python
EntropyOptimizer(..., seed_filter="mrmr", seed_size=8, initial_genes=[[0]])
EntropyOptimizer(..., seed_filter="boruta", seed_size=8, boruta_max_iter=30)

optimizer.seed_ranking   # e.g. [22, 16, 21, 13, 27, 10, 14, 23]
optimizer.seed_fits      # 0 for mRMR; Boruta's iterations otherwise

# The filters on their own
order = seeding.rank_mrmr(X_train, y_train, 5, random_state=0)
order, confirmed, fits = seeding.rank_boruta(DecisionTreeClassifier, X_train, y_train,
                                             random_state=0, max_iter=30)
seeding.nested_prefixes(order, 3)   # [[22], [22, 16], [22, 16, 21]]
```

---

## 8. LLM guidance — [08](examples/08_llm_guidance.py)

A large language model guides the search through the feature prior, which is
asked once, and the subset proposer, which is asked every few generations.
[llm-guidance.md](llm-guidance.md) explains both. The script asks a model
through OpenRouter when `OPENROUTER_API_KEY` is set. Without a key it runs
labelled rule-based stand-ins.

```python
# The minimal client: one chat completion, the reply text back
def chat(system, user):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 4000,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    ...   # POST to https://openrouter.ai/api/v1/chat/completions
    return reply["choices"][0]["message"]["content"]

# Feature prior: one call, one score per feature
class LLMFeaturePrior:
    def __call__(self, ctx):                       # ctx: GuidanceContext
        reply = chat(PRIOR_INSTRUCTIONS, describe(ctx))
        given = parse_json(reply)["scores"]        # {"<index>": score, ...}
        ...                                        # -> array of n scores in [0, 1]

# Subset proposer: shows the front, takes the subsets back
class LLMSubsetProposer:
    def __call__(self, ctx):                       # ctx: ProposalContext
        front = [f"size {e['size']}: {e['features']} -> {e['score']:.4f}"
                 for e in ctx.front]
        reply = chat(PROPOSER_INSTRUCTIONS.format(k=3, max_size=10),
                     describe(ctx.guidance) + "
".join(front))
        return parse_json(reply)["subsets"]        # [[index, ...], ...]

context = GuidanceContext(
    n_features=n, feature_names=names,
    task="Classify a breast mass as malignant or benign from features of "
         "the cell nuclei in a fine-needle aspirate image.",
    class_names=["malignant", "benign"])

EntropyOptimizer(..., PriorGuidedMutationStrategy(1 / n),
                 guidance_context=context,
                 feature_prior=LLMFeaturePrior(), prior_weight=0.5,
                 prior_uses=("mating", "mutation", "initial"),
                 prior_initial_singles=5,
                 subset_proposer=LLMSubsetProposer(), proposer_every=5)

optimizer.prior_scores       # the LLM's scores
optimizer.feature_weights    # combined with mutual information
optimizer.proposal_log[0]    # {'generation': 5, 'proposed': [...], 'evaluated': [...], 'fits': 3}
```

The script runs three searches at 300 fits each: without guidance, with the
prior, and with the prior and the proposer.

---

## 9. Regression — [09](examples/09_regression.py)

See [regression.md](regression.md) for the explanation. The short version:

```python
class RegressionMating(IntersectionMatingWithWeightedRandomInformationGain):
    def use_data_information(self, train_data, train_target):
        self.scikit_information_gain = mutual_info_regression(
            train_data, train_target, random_state=0)

EntropyOptimizer(partial(DecisionTreeRegressor, min_samples_leaf=5),
                 X_train, y_train, RegressionEvaluator(X_val, y_val, "r2"), n,
                 RandomEveryoneWithEveryone(pool_size=3,
                                            mating_strategy=RegressionMating()),
                 RandomFlipMutationStrategy(1 / n), initial_genes=singles, config=...)
```

On Friedman #1 with 20 noise features, the front reaches exactly the five
relevant features at size 5.

---

## 10. Your own strategies — [10](examples/10_custom_strategies.py)

```python
class SizeCappedMutation(MutationStrategy):        # flips, then a hard size cap
    def mutate_internal(self, item): ...

class UniformCrossover(MatingStrategy):            # each gene from either parent
    def mate_internal(self, item1, item2):
        take_first = random_utils.random(len(item1.genes)) < 0.5
        return numpy.where(take_first, item1.genes, item2.genes).tolist()

class Tournament(MatingSelectionStrategy):         # best of k, pairs mated in turn
    def mating_pool(self, population): ...         # -> MatingPoolResult(pool, carry_over)
    def mate_pool(self, mating_pool, current_generation=0): ...

class NeighbourSwap(SwapStrategy):                 # swap for an adjacent column
    def select_removal(self, genes, changes): return int(changes[0][1])
    def select_insertions(self, genes, removed, count): ...
```

Draw random numbers through `random_utils` so a seeded run repeats, and keep
the classes at module level so they pickle.
