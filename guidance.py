"""Outside guidance for the search: a feature prior and a subset proposer.

Both are hooks the caller injects into `EntropyOptimizer`; FASTENER knows
nothing about where they come from (an LLM, domain knowledge, another
selector). Both are off by default, and with neither the search is unchanged.

* A **feature prior** is one non-negative relevance score per feature, given
  as an array or as a callable that receives a `GuidanceContext` (feature
  names, task description, class names, free metadata) and returns the
  scores. It is resolved once, before the search. It can guide three
  operators, chosen with `prior_uses`:

  - ``"mating"``: the mating strategy's information-gain vector (used to sort
    and to sample the genes a child gets back) becomes
    ``w = MI^(1 - lambda) * prior^lambda``;
  - ``"mutation"``: a `PriorGuidedMutationStrategy` adds features in
    proportion to w and drops them in proportion to how weak they are, at
    the same expected number of flips as the uniform mutation;
  - ``"initial"``: `prior_initial_singles` single-feature subsets are drawn
    from w, without replacement, and added to the initial population.

  ``lambda = 0`` leaves the mating weights equal to mutual information;
  ``lambda = 1`` uses the prior alone.

* A **subset proposer** is a callable that receives a `ProposalContext` --
  the generation, the current front (features and validation score per
  size), the fits spent and the budget -- and returns feature subsets to
  try. It is called every `proposer_every` generations; its subsets are
  evaluated like any candidate, out of the same fit budget, and join the
  population. Every call and what it returned is kept in
  `EntropyOptimizer.proposal_log`.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

import random_utils
from item import Genes, Item, MutationStrategy

PRIOR_USES = ("mating", "mutation", "initial")
# Each vector is scaled to a maximum of 1 and floored here before combining,
# so a feature with zero mutual information (or zero prior) is unlikely but
# not impossible: w = 0 for every feature the prior leaves out would make the
# prior a hard constraint rather than a bias.
WEIGHT_FLOOR = 1e-3


@dataclass
class GuidanceContext:
    """What a prior or a proposer may know about the problem."""
    n_features: int
    feature_names: List[str]
    task: Optional[str] = None
    class_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposalContext:
    """The state of the search, as shown to a subset proposer."""
    guidance: GuidanceContext
    generation: int
    # One entry per front size: {"features": [...], "score": float}.
    front: List[Dict[str, Any]]
    model_fits: int
    max_model_fits: Optional[int]
    feature_weights: Optional[List[float]] = None


FeaturePrior = Union[Sequence[float], np.ndarray,
                     Callable[[GuidanceContext], Sequence[float]]]
SubsetProposer = Callable[[ProposalContext], Sequence[Sequence[int]]]


def resolve_prior(prior: FeaturePrior, context: GuidanceContext) -> np.ndarray:
    """The prior as a validated float vector, calling it if it is a hook."""
    scores = prior(context) if callable(prior) else prior
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (context.n_features,):
        raise ValueError(f"feature prior has shape {scores.shape}, expected "
                         f"({context.n_features},)")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("feature prior must be finite and non-negative")
    return scores


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    top = v.max() if len(v) else 0.0
    v = v / top if top > 0 else np.ones_like(v)
    return np.maximum(v, WEIGHT_FLOOR)


def combine(information_gain: Sequence[float], prior: np.ndarray,
            prior_weight: float) -> np.ndarray:
    """w = MI^(1 - lambda) * prior^lambda, each scaled to max 1 and floored."""
    if not 0.0 <= prior_weight <= 1.0:
        raise ValueError(f"prior_weight must be in [0, 1], not {prior_weight}")
    return _unit(information_gain) ** (1.0 - prior_weight) * \
        _unit(prior) ** prior_weight


def sample_singles(weights: np.ndarray, count: int) -> List[List[int]]:
    """`count` distinct single features drawn in proportion to `weights`."""
    count = min(count, int(np.count_nonzero(weights)))
    if count <= 0:
        return []
    picks = random_utils.choices(len(weights), size=count, replace=False,
                                 p=weights / weights.sum())
    return [[int(i)] for i in picks]


class PriorGuidedMutationStrategy(MutationStrategy):
    """Random flips, biased by feature weights.

    With `prob` p and n features the uniform mutation flips each gene with
    probability p. Here an unselected feature i is switched on with
    probability p * n * w_i / sum(w), and a selected one switched off with
    p * n * v_i / sum(v), v = max(w) - w + floor: strong features are added
    more often and dropped less often. The expected number of flips per
    child stays n * p when the child's genes are a typical mix; each
    probability is capped at 1. Until `use_feature_weights` is called the
    weights are uniform and it behaves as `RandomFlipMutationStrategy`, draw
    for draw.
    """

    def __init__(self, prob: float = 1 / 100) -> None:
        self.prob = prob
        self.p_on: Optional[np.ndarray] = None
        self.p_off: Optional[np.ndarray] = None

    def use_feature_weights(self, weights: np.ndarray) -> None:
        w = np.asarray(weights, dtype=float)
        n = len(w)
        v = w.max() - w + WEIGHT_FLOOR
        self.p_on = np.minimum(self.prob * n * w / w.sum(), 1.0)
        self.p_off = np.minimum(self.prob * n * v / v.sum(), 1.0)

    def mutate_internal(self, item: "Item") -> Genes:
        genes = np.asarray(item.genes, dtype=bool)
        draw = random_utils.random(len(genes))
        if self.p_on is None:
            flip = draw < self.prob
        else:
            flip = np.where(genes, draw < self.p_off, draw < self.p_on)
        return (genes ^ flip).tolist()


def front_summary(front: Dict[int, Any]) -> List[Dict[str, Any]]:
    """The front as plain data: features and validation score per size."""
    return [{"size": int(size),
             "features": [int(i) for i in np.flatnonzero(item.genes)],
             "score": float(item.result.score)}
            for size, item in sorted(front.items())]


def proposals_to_genes(proposals: Sequence[Sequence[int]],
                       n_features: int) -> List[Genes]:
    """Valid, distinct, non-empty subsets as gene lists; the rest dropped."""
    out, seen = [], set()
    for subset in proposals or []:
        idx = sorted({int(i) for i in subset if 0 <= int(i) < n_features})
        if not idx or tuple(idx) in seen:
            continue
        seen.add(tuple(idx))
        genes = np.zeros(n_features, dtype=bool)
        genes[idx] = True
        out.append(genes.tolist())
    return out
