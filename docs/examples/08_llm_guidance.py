"""LLM guidance: a large language model steers FASTENER's search.

FASTENER takes guidance from an LLM at two points in the algorithm:

* **Feature prior**, asked once before the search. The LLM reads the
  feature names, the task and any descriptions, and scores how relevant
  each feature is likely to be. FASTENER combines those scores with mutual
  information as w = MI^(1 - lambda) * prior^lambda and uses w in mating,
  in mutation and to draw extra single-feature starting subsets.
* **Subset proposer**, asked every few generations. The LLM reads the
  current Pareto front (features and validation score per size) and
  proposes new subsets. FASTENER evaluates them like any other candidate,
  and they compete for the front.

FASTENER calls these as plain functions (`feature_prior=`,
`subset_proposer=`), so any model and any provider will do. This script
implements both with a minimal OpenRouter client (stdlib only):

    export OPENROUTER_API_KEY=...                       # required for the LLM
    export FASTENER_LLM_MODEL=anthropic/claude-sonnet-5.5   # optional
    python 08_llm_guidance.py

Without a key the script still runs, with rule-based stand-ins in place of
the LLM. It says so at the start, so its output is not mistaken for LLM
guidance.
"""
from _common import ValidationEvaluator, classification_data, front_table, run_folder

import json
import os
import re
import urllib.request

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

from fastener import Config, EntropyOptimizer
from guidance import GuidanceContext, PriorGuidedMutationStrategy
from item import (IntersectionMatingWithWeightedRandomInformationGain,
                  RandomEveryoneWithEveryone)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("FASTENER_LLM_MODEL", "anthropic/claude-sonnet-5.5")


# --------------------------------------------------------------- the LLM
def chat(system: str, user: str) -> str:
    """One chat completion; returns the reply text."""
    body = {"model": MODEL, "temperature": 0, "max_tokens": 4000,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    request = urllib.request.Request(
        API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        reply = json.loads(response.read().decode("utf-8"))
    return reply["choices"][0]["message"]["content"] or ""


def parse_json(text: str):
    """The JSON object in a reply, with or without a code fence round it."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    text = fence.group(1) if fence else text
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0],
                default=-1)
    if start < 0:
        raise ValueError(f"no JSON in the reply: {text[:200]!r}")
    return json.JSONDecoder().raw_decode(text[start:])[0]


def describe(ctx: GuidanceContext) -> str:
    """The problem as the LLM sees it: task, classes, numbered features."""
    descriptions = ctx.metadata.get("feature_descriptions", {})
    lines = [f"Task: {ctx.task}"] if ctx.task else []
    if ctx.class_names:
        lines.append("Classes: " + ", ".join(ctx.class_names))
    lines.append(f"Features ({ctx.n_features}):")
    for i, name in enumerate(ctx.feature_names):
        extra = descriptions.get(name)
        lines.append(f"{i}: {name}" + (f" -- {extra}" if extra else ""))
    return "\n".join(lines)


PRIOR_INSTRUCTIONS = (
    "You are helping a feature-selection search. Rate how useful each "
    "feature is likely to be for the task, on its own or with a few others, "
    "from 0 (useless) to 1 (most useful). You do not see the data; use your "
    "knowledge of the domain. Reply with JSON only: "
    '{"scores": {"<feature index>": <score>, ...}}, one entry per feature.')

PROPOSER_INSTRUCTIONS = (
    "You are helping a feature-selection search. Below is its current best "
    "subset for each size, with the validation score (higher is better). "
    "Propose up to {k} new subsets of at most {max_size} features that you "
    "expect to beat the current subsets of the same size. Use feature "
    'indices. Reply with JSON only: {{"subsets": [[<index>, ...], ...]}}.')


class LLMFeaturePrior:
    """Feature prior: one LLM call, one relevance score per feature."""

    def __init__(self):
        self.reply = None      # kept for the record

    def __call__(self, ctx: GuidanceContext):
        self.reply = chat(PRIOR_INSTRUCTIONS, describe(ctx))
        given = parse_json(self.reply).get("scores", {})
        scores = np.full(ctx.n_features, np.nan)
        for key, value in given.items():
            try:
                i, x = int(key), float(value)
            except (TypeError, ValueError):
                continue                     # skip what is not index: number
            if 0 <= i < ctx.n_features and np.isfinite(x):
                scores[i] = min(max(x, 0.0), 1.0)
        # A feature the LLM left out gets the lowest score it gave any.
        known = ~np.isnan(scores)
        scores[~known] = scores[known].min() if known.any() else 1.0
        return scores


class LLMSubsetProposer:
    """Subset proposer: shows the LLM the front, takes its subsets."""

    def __init__(self, k=3, max_size=10):
        self.k, self.max_size = k, max_size
        self.replies = []

    def __call__(self, ctx):
        front = [f"size {e['size']}: {e['features']} -> {e['score']:.4f}"
                 for e in ctx.front]
        user = (describe(ctx.guidance) + f"\n\nGeneration {ctx.generation}, "
                f"{ctx.model_fits} model fits spent.\nCurrent front:\n"
                + "\n".join(front))
        reply = chat(PROPOSER_INSTRUCTIONS.format(k=self.k,
                                                  max_size=self.max_size), user)
        self.replies.append(reply)
        try:
            subsets = parse_json(reply).get("subsets", [])
        except (ValueError, AttributeError):
            return []                        # an unusable reply proposes nothing
        out = []
        for subset in subsets[:self.k]:
            # Keep the integer entries; FASTENER itself drops out-of-range
            # indices, empty subsets and duplicates.
            ints = [int(i) for i in subset
                    if isinstance(i, (int, float)) or str(i).isdigit()]
            out.append(ints[:self.max_size])
        return out


# ----------------------------------------------- stand-ins without a key
class RuleBasedPrior:
    """Not an LLM: scores features by keywords, to run without a key."""

    def __call__(self, ctx):
        keywords = ctx.metadata.get("keywords", [])
        return [1.0 if any(k in name for k in keywords) else 0.1
                for name in ctx.feature_names]


class RuleBasedProposer:
    """Not an LLM: extends the best subset by its highest-weight missing
    features, remembering what it has tried (a repeat would cost nothing,
    since FASTENER caches every subset it has fitted, but add nothing)."""

    def __init__(self, k=3):
        self.k, self.tried = k, set()

    def __call__(self, ctx):
        best = max(ctx.front, key=lambda entry: entry["score"])
        weights = np.asarray(ctx.feature_weights
                             or np.ones(ctx.guidance.n_features))
        out = []
        for i in np.argsort(-weights):
            subset = tuple(sorted(set(best["features"]) | {int(i)}))
            if int(i) in best["features"] or subset in self.tried:
                continue
            self.tried.add(subset)
            out.append(list(subset))
            if len(out) == self.k:
                break
        return out


# --------------------------------------------------------------- the run
def main():
    X_train, y_train, X_val, y_val, _, _, names = classification_data()
    n = X_train.shape[1]
    use_llm = bool(os.environ.get("OPENROUTER_API_KEY"))
    print(f"LLM guidance with {MODEL}" if use_llm else
          "OPENROUTER_API_KEY is not set: running RULE-BASED STAND-INS, "
          "not an LLM")

    # What the LLM is told about the problem. `metadata` can carry feature
    # descriptions (a data dictionary); `keywords` is for the stand-in only.
    context = GuidanceContext(
        n_features=n, feature_names=names,
        task="Classify a breast mass as malignant or benign from features "
             "of the cell nuclei in a fine-needle aspirate image.",
        class_names=["malignant", "benign"],
        metadata={"keywords": ["concave", "radius", "texture"]})

    source = "llm" if use_llm else "rule_based"
    prior = LLMFeaturePrior() if use_llm else RuleBasedPrior()
    proposer = LLMSubsetProposer() if use_llm else RuleBasedProposer()

    def run(name, **guidance):
        optimizer = EntropyOptimizer(
            DecisionTreeClassifier, X_train, y_train,
            ValidationEvaluator(X_val, y_val, balanced_accuracy_score), n,
            RandomEveryoneWithEveryone(
                pool_size=3,
                mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
            # Needed for prior_uses "mutation"; without a prior it is the
            # plain random flip.
            PriorGuidedMutationStrategy(1 / n),
            initial_genes=[[i] for i in range(n)],
            guidance_context=context,
            config=Config(output_folder=run_folder(name), random_seed=4,
                          number_of_rounds=10_000, max_model_fits=300,
                          dump_generation_logs=False),
            **guidance)
        optimizer.mainloop()
        print(f"\n{name}:\n" + front_table(optimizer, names))
        return optimizer

    # 1. No guidance: the base algorithm, for comparison.
    run("no_guidance")

    # 2. Feature prior. prior_weight 0 is mutual information alone, 1 the
    #    LLM's scores alone. prior_uses picks what the weights steer.
    opt = run(f"{source}_prior", feature_prior=prior, prior_weight=0.5,
              prior_uses=("mating", "mutation", "initial"),
              prior_initial_singles=5)
    print("prior's top 5:", [names[i] for i in np.argsort(-opt.prior_scores)[:5]])
    print("combined top 5:", [names[i] for i in np.argsort(-opt.feature_weights)[:5]])

    # 3. Feature prior and subset proposer together.
    opt = run(f"{source}_prior_and_proposer", feature_prior=prior,
              subset_proposer=proposer, proposer_every=5)
    for entry in opt.proposal_log:
        print(f"  gen {entry['generation']:3d}: proposed {entry['proposed']}, "
              f"{entry['fits']} new fits, scores "
              f"{[round(e['score'], 4) for e in entry['evaluated']]}")


if __name__ == "__main__":
    main()
