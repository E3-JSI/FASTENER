# LLM guidance

FASTENER can be guided by a large language model. Mutual information and
permutation importance only see the data. An LLM brings what the data cannot
show: what the features mean, and what is known about the task. It enters
the algorithm at two points, the **feature prior** and the **subset
proposer**. Both are off by default, and they can be used alone or together.

The LLM never scores a subset. Every score on the front still comes from
fitting your model and running your evaluator. A poor suggestion costs fits,
but it cannot make a subset look better than it scores.

The complete, runnable example is
[examples/08_llm_guidance.py](examples/08_llm_guidance.py). It contains a
minimal OpenRouter client (standard library only), the two prompts, and the
reply parsing.

```bash
export OPENROUTER_API_KEY=...
export FASTENER_LLM_MODEL=anthropic/claude-sonnet-5.5   # optional; any OpenRouter model ID
python 08_llm_guidance.py
```

Without a key the script runs rule-based stand-ins in place of the LLM, and
says so on its first line.

## What the LLM is told: `GuidanceContext`

```python
from guidance import GuidanceContext

context = GuidanceContext(
    n_features=n,
    feature_names=names,                     # what the LLM reasons about
    task="Classify a breast mass as malignant or benign from features of "
         "the cell nuclei in a fine-needle aspirate image.",
    class_names=["malignant", "benign"],
    metadata={"feature_descriptions": {"worst radius": "largest nucleus radius"},
              "dataset_description": "..."})
EntropyOptimizer(..., guidance_context=context)
```

Without a context, the features are called `f0, f1, …` and there is no task.
An LLM can do little with that. Meaningful names and a one-line task
description are what make the guidance worth having. `metadata` is free-form,
and your prior and proposer decide what to show the model.

## The feature prior

**What the LLM does.** It is asked once, in `prepare_loop` before the first
generation, to score how relevant each feature is likely to be (0 to 1). The
example's prompt:

> You are helping a feature-selection search. Rate how useful each feature is
> likely to be for the task, on its own or with a few others, from 0 (useless)
> to 1 (most useful). You do not see the data; use your knowledge of the
> domain. Reply with JSON only: `{"scores": {"<feature index>": <score>, ...}}`

**What FASTENER does with it.** It combines the scores with mutual information
into one weight per feature:

```
w = unit(MI) ** (1 - prior_weight) * unit(prior) ** prior_weight
```

`unit` scales each vector to a maximum of 1 and floors it at 0.001, so a
feature the LLM dismisses becomes unlikely but stays reachable. `prior_uses`
chooses where `w` acts:

| Use | Effect |
|---|---|
| `"mating"` | `w` replaces mutual information in the crossover: it decides which of the parents' differing features a child gets. Needs an information-gain mating strategy. |
| `"mutation"` | `PriorGuidedMutationStrategy` switches on high-weight features and switches off low-weight ones more often, at the same average number of flips. Needs that mutation strategy. |
| `"initial"` | `prior_initial_singles` single features are drawn by `w` and added to the starting population. |

```python
from guidance import PriorGuidedMutationStrategy

EntropyOptimizer(
    model, X_train, y_train, evaluator, n, mating,
    PriorGuidedMutationStrategy(1 / n),
    initial_genes=[[i] for i in range(n)],
    guidance_context=context,
    feature_prior=LLMFeaturePrior(),       # any callable: GuidanceContext -> scores
    prior_weight=0.5,                      # 0: MI only; 1: the LLM only
    prior_uses=("mating", "mutation", "initial"),
    prior_initial_singles=5)
```

The prior may also be an array of scores. One LLM call can then be saved to
a file and reused across runs and seeds, which also makes the runs
reproducible.

## The subset proposer

**What the LLM does.** Every `proposer_every` generations (default 5), after
the front is updated, it sees the problem description and the current front,
and proposes subsets. The example's prompt:

> You are helping a feature-selection search. Below is its current best subset
> for each size, with the validation score (higher is better). Propose up to 3
> new subsets of at most 10 features that you expect to beat the current
> subsets of the same size. Use feature indices. Reply with JSON only:
> `{"subsets": [[<index>, ...], ...]}`

The proposer receives a `ProposalContext`:

| Field | Contents |
|---|---|
| `guidance` | The `GuidanceContext` above |
| `generation` | The current generation |
| `front` | `[{"size", "features", "score"}, ...]`, one entry per size |
| `model_fits`, `max_model_fits` | Fits spent and the budget (`None` without one) |
| `feature_weights` | `w`, if there is a prior; otherwise `None` |

**What FASTENER does with it.** It drops out-of-range indices, empty subsets
and duplicates, then evaluates each remaining subset with your model, from
the same fit budget, and adds it to the population. A proposal that beats the
front entry of its size replaces it.

```python
EntropyOptimizer(..., guidance_context=context,
                 subset_proposer=LLMSubsetProposer(), proposer_every=5)
```

A subset FASTENER has already fitted costs nothing to propose again, and adds
nothing. Show the LLM what was tried, or remember it in the proposer.

## What a run records

| Attribute | Contents |
|---|---|
| `optimizer.prior_scores` | The prior as returned |
| `optimizer.feature_weights` | `w`, the combined weights |
| `optimizer.prior_singles` | The single features drawn for the start |
| `optimizer.proposal_log` | Per call: `generation`, `proposed`, `evaluated` (features and score), `fits` |

The optimizer's pickle (`experiment.pickle`) leaves out the prior and proposer
objects, which may hold network clients, and keeps their results. Keep the
LLM's raw replies yourself if you need them: the example's classes store them
in `.reply` and `.replies`.

## Reproducibility and cost

An LLM call is not deterministic across providers or model versions, even at
temperature 0. To repeat a guided run exactly, record each request and reply
and replay them. A record-and-replay log keyed by a hash of the request does
this, and because FASTENER's search is deterministic for a seed, a replayed
run asks the same questions in the same order. The cost is one call for the
prior (or a few, in chunks, for thousands of features), plus one call per
proposer round.

## Control runs

To tell whether the LLM's *knowledge* helps, rather than the extra
candidates or the changed weights, compare against:

* the same run without guidance;
* a **scrambled** prior: the LLM sees neutral names (`feature_00`, …) instead
  of the real ones, with the task and classes kept;
* a proposer that returns random subsets of the same sizes.

Compare at an equal `max_model_fits`, over several seeds, on held-out data.

## Without an LLM

The two arguments take any callable or array, so the same mechanism carries
other guidance: a domain expert's ranking, scores from another selector, or a
rule. The example's stand-ins (`RuleBasedPrior`, `RuleBasedProposer`) are
two such rules.
