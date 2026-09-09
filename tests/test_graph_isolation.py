'''
Tests that each Chronos instance owns its tensorflow graph, and that building many models in one
process does not accumulate graph nodes or leak graphs.

Background: tensorflow's implicit default graph is per-thread (the stack supplying it subclasses
`threading.local`), so threads were never the problem. The problem is that every Chronos built in
the same thread shared that thread's graph, and a `tf.Graph` cannot have nodes removed from it.
Before the fix, two models from identical inputs left 9606 nodes on one shared graph instead of
4818 on each of two, and repeatedly building a model grew RSS without bound.

Run the slow memory tests with `pytest -m slow`, or skip them with `pytest -m "not slow"`.
'''
import gc
import inspect
import os

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import chronos
from chronos.hit_calling import ConditionComparison


DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
SAMPLE = os.path.join(DATA, "SampleData")


def _live_graphs():
    '''Non-empty live tf.Graph objects. Empty ones are excluded because merely calling
    `tf.compat.v1.get_default_graph()` creates an empty per-thread graph that lives forever.'''
    gc.collect()
    return [g for g in gc.get_objects()
            if isinstance(g, tf.Graph) and len(g.get_operations()) > 0]


@pytest.fixture(scope="module")
def inputs():
    '''A two-library slice of the sample data, small enough to build in a couple of seconds.

    Do not shrink this much further. Chronos needs negative controls to estimate the
    overdispersion parameter, and below roughly 80 genes `smart_initialize` fails with
    "estimated efficacy outside bounds" -- on unmodified main as well, so it is a property of the
    fixture rather than of the code under test.
    '''
    nonessentials = set(pd.read_csv(os.path.join(SAMPLE, "AchillesNonessentialControls.csv"))["Gene"])
    raw = {}
    for lib, tag in [("avana", "Avana"), ("ky", "KY")]:
        raw[lib] = (
            chronos.read_hdf5(os.path.join(SAMPLE, "%sReadcounts.hdf5" % tag)),
            pd.read_csv(os.path.join(SAMPLE, "%sGuideMap.csv" % tag)),
            pd.read_csv(os.path.join(SAMPLE, "%sSequenceMap.csv" % tag)),
        )

    shared = set(raw["avana"][1].gene) & set(raw["ky"][1].gene)
    controls = sorted(shared & nonessentials)[:150]
    genes = sorted(shared - nonessentials)[:400] + controls

    readcounts, guide_gene_map, sequence_map, negative_control_sgrnas = {}, {}, {}, {}
    for lib, (rc, gm, sm) in raw.items():
        gm = gm[gm.gene.isin(genes) & gm.sgrna.isin(rc.columns)]
        guide_gene_map[lib] = gm
        readcounts[lib] = rc[sorted(set(gm.sgrna))]
        sequence_map[lib] = sm
        negative_control_sgrnas[lib] = gm.sgrna[gm.gene.isin(controls)]

    return dict(
        readcounts=readcounts,
        guide_gene_map=guide_gene_map,
        sequence_map=sequence_map,
        negative_control_sgrnas=negative_control_sgrnas,
        print_to=None,
    )


def test_build_and_init_signatures_match():
    '''`__init__` forwards to `_build` via `locals()`, so the two parameter lists must agree.'''
    init = inspect.signature(chronos.Chronos.__init__).parameters
    build = inspect.signature(chronos.Chronos._build).parameters
    assert list(init) == list(build)


def test_each_model_owns_its_graph(inputs):
    first = chronos.Chronos(**inputs)
    second = chronos.Chronos(**inputs)
    try:
        assert first.graph is not second.graph
        assert first.sess.graph is first.graph
        assert second.sess.graph is second.graph
        # identical inputs must give identical graph sizes. Sharing a graph would make the second
        # model's count the running total instead (4818 -> 9606 before the fix).
        assert len(first.graph.get_operations()) == len(second.graph.get_operations())
    finally:
        del first, second


def test_nothing_is_built_on_the_default_graph(inputs):
    model = chronos.Chronos(**inputs)
    try:
        assert len(tf.compat.v1.get_default_graph().get_operations()) == 0
        assert len(model.graph.get_operations()) > 0
    finally:
        del model


def test_persistent_handles_is_gone(inputs):
    '''`persistent_handles` existed only to feed a `delete_session_tensor` loop in `__del__` that
    built ops instead of freeing anything. Both are gone; closing the session frees the tensors.'''
    assert not hasattr(chronos.Chronos, "persistent_handles")
    model = chronos.Chronos(**inputs)
    try:
        assert not hasattr(model, "persistent_handles")
    finally:
        del model


def test_snapshot_works_for_a_second_model(inputs, tmp_path):
    '''`summary.merge_all()` collects the graph-wide SUMMARIES collection. While models shared a
    graph, the second model's `_merged` pulled in the first model's summary ops, whose placeholders
    are absent from the second model's `run_dict`, so `snapshot()` raised InvalidArgumentError.'''
    first = chronos.Chronos(log_dir=str(tmp_path / "log1"), **inputs)
    second = chronos.Chronos(log_dir=str(tmp_path / "log2"), **inputs)
    try:
        second.train(2)
        second.snapshot()
    finally:
        del first, second


@pytest.mark.slow
def test_repeated_builds_do_not_retain_graphs(inputs):
    '''The sharp assertion for the leak: dropping a model must leave no graph behind.'''
    for _ in range(4):
        model = chronos.Chronos(**inputs)
        model.train(5)
        del model
        assert _live_graphs() == []


@pytest.mark.slow
def test_repeated_builds_plateau_in_memory(inputs):
    '''Assert a plateau, not a decline: tensorflow's allocator does not return freed pages to the
    OS, so RSS creeps up slightly even with no leak. Before the fix this loop roughly doubled RSS
    (862 -> 1998 MB over eight iterations) and kept climbing.'''
    psutil = pytest.importorskip("psutil")
    process = psutil.Process()
    rss = []
    for _ in range(8):
        model = chronos.Chronos(**inputs)
        model.train(5)
        del model
        gc.collect()
        rss.append(process.memory_info().rss / 1e6)

    growth = rss[-1] - rss[-4]
    assert growth < 0.1 * rss[0], "RSS still climbing over the last few iterations: %r" % rss


def _condition_comparison():
    nonessentials = pd.read_csv(os.path.join(SAMPLE, "AchillesNonessentialControls.csv"))["Gene"]
    guide_gene_map = pd.read_csv(os.path.join(SAMPLE, "DeWeirdtGuideMap.csv"))
    return ConditionComparison(
        readcounts={"brunello": chronos.read_hdf5(os.path.join(SAMPLE, "DeWeirdtReadcounts.hdf5"))},
        condition_map={"brunello": pd.read_csv(os.path.join(SAMPLE, "DeWeirdtConditionMap.csv"))},
        guide_gene_map={"brunello": guide_gene_map},
        negative_control_sgrnas={"brunello": guide_gene_map.sgrna[
            guide_gene_map.gene.isin([s.split(' ')[0] for s in nonessentials])]},
        print_to=None,
    )


def test_comparisons_do_not_share_a_scratch_directory():
    '''The scratch directory used to be the fixed relative path
    ".chronos_compare_undistinguished_model", so two comparisons in one working directory
    overwrote each other's saved model.'''
    first, second = _condition_comparison(), _condition_comparison()
    try:
        assert first._get_scratch_dir() != second._get_scratch_dir()
        assert os.path.isdir(first._get_scratch_dir())
    finally:
        first._clear_scratch_dir()
        second._clear_scratch_dir()
    assert first._scratch_dir is None


@pytest.mark.slow
def test_compare_conditions_cleans_up(tmp_path, monkeypatch):
    '''`compare_conditions` trains four models in one process, which is the loop that made the
    leak painful. It must finish, leave no scratch directory, and retain no graph.'''
    monkeypatch.chdir(tmp_path)
    comparison = _condition_comparison()
    statistics = comparison.compare_conditions(
        ("Control", "A-1331852"), max_null_iterations=2, nepochs=30
    )
    assert len(statistics) > 0
    assert comparison._scratch_dir is None
    del comparison
    assert _live_graphs() == []


@pytest.mark.slow
def test_alternate_cn_does_not_touch_the_default_graph():
    '''`add_global_shift` built a spline model and a session per cell line group, on the default
    graph, and never closed the session. Four groups left 632 ops behind before the fix.'''
    gene_effect = chronos.read_hdf5(os.path.join(DATA, "Achilles_run/gene_effect.hdf5"))
    copy_number = chronos.read_hdf5(os.path.join(SAMPLE, "OmicsCNGene.hdf5"))
    lines = sorted(set(gene_effect.index) & set(copy_number.index))
    genes = sorted(set(gene_effect.columns) & set(copy_number.columns))[:150]
    gene_effect = gene_effect.loc[lines, genes]
    copy_number = copy_number.loc[lines, genes]

    before = len(tf.compat.v1.get_default_graph().get_operations())
    # max_lines below the line count forces several groups, i.e. several calls
    adjusted, _ = chronos.alternate_CN(gene_effect, copy_number, max_lines=max(3, len(lines) // 3))
    assert adjusted.shape == gene_effect.shape
    assert len(tf.compat.v1.get_default_graph().get_operations()) == before
