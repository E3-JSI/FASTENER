"""Tests for outside guidance: the feature prior and the subset proposer."""
import pickle

import numpy as np
import pytest

from conftest import N_FEATURES, item_from, selected

import guidance
import random_utils
from fastener import Config, EntropyOptimizer
from guidance import GuidanceContext, PriorGuidedMutationStrategy
from item import (IntersectionMatingWithWeightedRandomInformationGain,
                  RandomEveryoneWithEveryone, RandomFlipMutationStrategy)

from test_optimizer import AccuracyEvaluator, CountingModelFactory


def guided_optimizer(small_data, name, mutation=None, **kwargs):
    X, y, X_val, y_val = small_data
    config = {"number_of_rounds": 6, "reset_to_pareto_rounds": 2,
              **kwargs.pop("config", {})}
    factory = CountingModelFactory()
    optimizer = EntropyOptimizer(
        factory, X, y, AccuracyEvaluator(X_val, y_val), N_FEATURES,
        RandomEveryoneWithEveryone(
            pool_size=3,
            mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
        mutation or PriorGuidedMutationStrategy(1 / N_FEATURES),
        initial_genes=kwargs.pop("initial_genes", [[5]]),
        config=Config(name, 2020, **config),
        **kwargs)
    optimizer.factory = factory
    return optimizer


# ------------------------------------------------------------------ the prior
def test_combine_ends_and_bounds():
    mi = np.array([0.0, 0.5, 1.0])
    prior = np.array([1.0, 0.0, 0.5])
    np.testing.assert_allclose(guidance.combine(mi, prior, 0.0),
                               [guidance.WEIGHT_FLOOR, 0.5, 1.0])
    np.testing.assert_allclose(guidance.combine(mi, prior, 1.0),
                               [1.0, guidance.WEIGHT_FLOOR, 0.5])
    with pytest.raises(ValueError):
        guidance.combine(mi, prior, 1.5)


def test_resolve_prior_calls_the_hook_with_the_context():
    seen = {}

    def hook(ctx):
        seen["ctx"] = ctx
        return [1.0] * ctx.n_features

    ctx = GuidanceContext(3, ["a", "b", "c"], task="t")
    np.testing.assert_array_equal(guidance.resolve_prior(hook, ctx), [1, 1, 1])
    assert seen["ctx"].feature_names == ["a", "b", "c"]
    with pytest.raises(ValueError):
        guidance.resolve_prior([1.0, 2.0], ctx)
    with pytest.raises(ValueError):
        guidance.resolve_prior([1.0, -1.0, 0.0], ctx)


def test_prior_reaches_mating_mutation_and_initial(small_data, run_dir):
    prior = np.full(N_FEATURES, 0.01)
    prior[3] = 1.0
    opt = guided_optimizer(small_data, "all_uses", feature_prior=prior,
                           prior_weight=1.0, prior_initial_singles=2)
    opt.prepare_loop()
    np.testing.assert_allclose(opt.prior_scores, prior)
    # lambda = 1: the weights are the prior's, scaled to max 1.
    np.testing.assert_allclose(opt.feature_weights, prior)
    strategy = opt.mating_selection_strategy.mating_strategy
    np.testing.assert_allclose(strategy.scikit_information_gain, prior)
    assert opt.mutation_strategy.p_on[3] > opt.mutation_strategy.p_on[0]
    assert len(opt.prior_singles) == 2
    sizes = {s for s, items in opt.population.items() for _ in items}
    assert sizes == {1}
    starts = [selected(i.genes) for items in opt.population.values() for i in items]
    assert {5} in starts and all(s in starts for s in map(set, opt.prior_singles))


def test_prior_hook_sees_names_and_task(small_data, run_dir):
    calls = []
    ctx = GuidanceContext(N_FEATURES, [f"x{i}" for i in range(N_FEATURES)],
                          task="sum sign", class_names=["neg", "pos"])
    opt = guided_optimizer(small_data, "hook",
                           feature_prior=lambda c: calls.append(c) or [1.0] * c.n_features,
                           guidance_context=ctx)
    opt.mainloop()
    assert len(calls) == 1 and calls[0].class_names == ["neg", "pos"]
    # The optimizer pickles although the hook (a lambda) does not.
    state = pickle.loads(pickle.dumps(opt))
    assert state.feature_prior is None
    np.testing.assert_allclose(state.prior_scores, 1.0)


def test_mutation_use_needs_a_weighted_strategy(small_data, run_dir):
    opt = guided_optimizer(small_data, "no_weights",
                           mutation=RandomFlipMutationStrategy(1 / N_FEATURES),
                           feature_prior=np.ones(N_FEATURES))
    with pytest.raises(ValueError, match="mutation"):
        opt.prepare_loop()
    opt2 = guided_optimizer(small_data, "no_weights_ok",
                            mutation=RandomFlipMutationStrategy(1 / N_FEATURES),
                            feature_prior=np.ones(N_FEATURES),
                            prior_uses=("mating", "initial"))
    opt2.prepare_loop()


def test_bad_options_are_refused(small_data, run_dir):
    with pytest.raises(ValueError):
        guided_optimizer(small_data, "bad1", prior_uses=("crossover",))
    with pytest.raises(ValueError):
        guided_optimizer(small_data, "bad2", prior_weight=-0.1)
    with pytest.raises(ValueError):
        guided_optimizer(small_data, "bad3", proposer_every=0)


# ------------------------------------------------------------------ mutation
def test_guided_mutation_without_weights_is_the_uniform_one():
    item = item_from([0, 2])
    random_utils.seed(7)
    a = [RandomFlipMutationStrategy(0.3).mutate_internal(item) for _ in range(20)]
    random_utils.seed(7)
    b = [PriorGuidedMutationStrategy(0.3).mutate_internal(item) for _ in range(20)]
    assert a == b


def test_guided_mutation_adds_strong_and_drops_weak_features():
    w = np.full(N_FEATURES, 0.1)
    w[1] = 1.0
    m = PriorGuidedMutationStrategy(1 / N_FEATURES)
    m.use_feature_weights(w)
    assert m.p_on[1] > m.p_on[0]
    assert m.p_off[1] < m.p_off[0]
    # The uniform rate on average, over all genes.
    assert m.p_on.mean() == pytest.approx(1 / N_FEATURES)
    random_utils.seed(1)
    added = np.zeros(N_FEATURES)
    for _ in range(4000):
        genes = m.mutate_internal(item_from([]))
        added += np.asarray(genes, dtype=float)
    assert added[1] > 3 * added[0]


# ------------------------------------------------------------------ proposer
def test_proposer_runs_every_n_rounds_and_pays_from_the_budget(small_data, run_dir):
    contexts = []

    def proposer(ctx):
        contexts.append(ctx)
        return [[0, 1], [0, 1], [99], [], [2, 3, 4]]

    opt = guided_optimizer(small_data, "proposer", subset_proposer=proposer,
                           proposer_every=2,
                           config={"number_of_rounds": 6})
    opt.mainloop()
    assert [c.generation for c in contexts] == [2, 4, 6]
    assert all(c.front and "features" in c.front[0] for c in contexts)
    first = opt.proposal_log[0]
    # Duplicates, out-of-range indices and empty subsets are dropped.
    assert [e["features"] for e in first["evaluated"]] == [[0, 1], [2, 3, 4]]
    assert first["fits"] <= 2
    # Features 0 and 1 decide the label: the proposal reaches the front.
    assert {0, 1} in [selected(i.genes) for i in opt.pareto_front.values()]
    assert opt.model_fits == opt.factory.fits


def test_proposer_stops_at_the_fit_budget(small_data, run_dir):
    opt = guided_optimizer(
        small_data, "proposer_budget",
        subset_proposer=lambda ctx: [[i] for i in range(N_FEATURES)] +
        [[i, j] for i in range(N_FEATURES) for j in range(i + 1, N_FEATURES)],
        proposer_every=1, config={"number_of_rounds": 50, "max_model_fits": 15})
    opt.mainloop()
    assert opt.stop_reason == "fit_budget"
    assert opt.model_fits == 15
    assert sum(e["fits"] for e in opt.proposal_log) <= 15
    pickle.loads(pickle.dumps(opt))
