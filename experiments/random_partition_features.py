"""
Random partition of matched granularity: the E2 control from
local-docs/experimental_design.md --

    "The central threat to your claim is that leaf-index recycling is a
    massively lossy compression, so degradation may follow from the
    data-processing inequality alone rather than from anything
    collapse-specific. Control: replace the RF's learned leaf partition with
    a random partition of matched granularity (same number of cells, same
    average cell occupancy, but not learned from the data) and run the
    identical loop. If both degrade identically, you have found generic
    information loss. If the learned-partition loop degrades differently --
    faster, or with a distinct tail-loss signature -- you have isolated the
    RF-specific mechanism. [...] report I(X;L) (metric 7) alongside, so
    degradation can be plotted against information retained."

This is a THIRD feature-generation mechanism, not a variant of the other
two: synthetic_features.py's OOB predictions and leaf_index_features.py's
leaf indices are both learned from (X, y); this module's partitions are
deliberately NOT -- cell membership is drawn uniformly at random, and X, y
are used only to measure a real fitted forest's granularity to match, never
to decide which sample goes in which cell. That is what makes I(X;L)
(experiments.metrics.mutual_information_x_leaf) computed on these ids the
low-information anchor point on E2's degradation-vs-information plot: a
real forest's leaves are a deterministic function of X (I(X;L) > 0 by
construction), a random partition of the same shape is not -- though the
measured I(X;L) will sit near the estimator's finite-sample bias floor
rather than exactly 0 (mutual_info_score is upward-biased whenever category
counts are large relative to n, regardless of true independence); the
comparison that matters is that it is far below the real leaves' I(X;L),
not that it is literally zero. See this module's __main__ demo.

"Matched granularity" is measured, not assumed: real trees don't produce
n_cells = n / nodesize exactly, and leaf sizes within one tree are rarely
equal. cell_sizes_from_leaf_ids() reads the ACTUAL per-tree leaf-size
multiset off a real fitted forest (leaf_index_features.fit_leaf_forest's
leaf_ids), and random_partition_ids() reproduces that multiset's cell COUNT
per tree and occupancy-distribution SHAPE, not merely its average, with
uniformly random membership -- strictly more than the doc's stated "same
number of cells, same average cell occupancy," which only makes the
"matched granularity" claim more defensible, not less. Because the fixed
real test set is rarely the same size as the data the real forest was fit
on, _rescale_cell_sizes() rescales the measured multiset onto whatever row
count is being partitioned (train or query), preserving proportional shape
rather than assuming n stays constant.

A random partition cannot generalize to held-out rows the way a real
fitted tree can -- membership was never a function of X, so there is no
rule to apply to new data. transform_random_partition() therefore draws a
FRESH, independent, equally-matched-granularity partition for query data
(e.g. the fixed real test set) rather than "transferring" anything from
training -- which is the correct behavior here, not a limitation to work
around: cell identity itself carries no information in this control, so
there is nothing meaningful to transfer.

Encoding (ordinal/one-hot) is delegated to leaf_index_features.py's
leaf_features/ordinal_leaf_features/onehot_leaf_features -- encoding an
(n, ntree) integer id matrix is identical work whether those ids came from
a real tree or a random draw, so it is not reimplemented here.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]

from ._common import child_seeds
from .leaf_index_features import LeafIndexConfig, VALID_ENCODINGS, fit_leaf_forest, leaf_features


def cell_sizes_from_leaf_ids(leaf_ids):
    """Per-tree leaf-SIZE multiset (not just count) from a real fitted
    forest's leaf ids -- the "matched granularity" random_partition_ids
    should reproduce. list of length n_trees, each entry a 1-D array of
    that tree's leaf occupancy counts."""
    return [np.unique(leaf_ids[:, t], return_counts=True)[1]
            for t in range(leaf_ids.shape[1])]


def _rescale_cell_sizes(sizes, n_target):
    """Rescale a per-tree cell-size multiset to sum to exactly `n_target`,
    preserving each cell's relative share of the total as closely as integer
    rounding allows (largest-remainder method).

    Needed because "matched granularity" has to apply to data of a
    DIFFERENT row count than the real forest was fit on -- the fixed real
    test set is rarely the same size as the training set it came from (Tier
    A's default is 10k train / 20k test). Rescaling the real forest's
    occupancy-distribution SHAPE onto n_target rows matches proportional
    cell count and average occupancy exactly, and the full distribution
    shape as closely as integer cell sizes allow -- a strictly better match
    than falling back to equal-sized cells just because n changed.
    """
    if sizes.sum() == n_target:
        return sizes
    shares = sizes * (n_target / sizes.sum())
    floors = np.floor(shares).astype(np.int64)
    remainder = n_target - floors.sum()
    frac_order = np.argsort(-(shares - floors))
    floors[frac_order[:remainder]] += 1
    return floors


def _random_partition_column(n, cell_sizes, rng):
    """One tree's worth of random partition ids over n rows: exactly
    `cell_sizes` cells (already rescaled to sum to n), membership uniformly
    random and independent of X/y."""
    order = rng.permutation(n)
    groups = np.split(order, np.cumsum(cell_sizes)[:-1])
    col = np.empty(n, dtype=np.int64)
    for cell_id, members in enumerate(groups):
        col[members] = cell_id
    return col


def random_partition_ids(n_samples, cell_sizes_per_tree, random_state):
    """(n_samples, n_trees) int array, the random-partition analog of a real
    forest's clf.apply(X): same per-tree cell COUNT and (rescaled to
    n_samples, see _rescale_cell_sizes) occupancy-distribution shape as
    `cell_sizes_per_tree`, but membership assigned uniformly at random,
    independent of X and y."""
    rng = np.random.default_rng(random_state)
    columns = [
        _random_partition_column(n_samples, _rescale_cell_sizes(sizes, n_samples), rng)
        for sizes in cell_sizes_per_tree
    ]
    return np.column_stack(columns)


def recycle_random_partition(X, y, config=LeafIndexConfig(), random_state=0,
                              mode="replace", encoding="onehot", leaf_ids=None):
    """Produce the next generation's training features via a random
    partition of matched granularity -- the E2 counterpart to
    leaf_index_features.recycle_leaf_indices. Same
    (X, y, config, random_state, mode, encoding) -> (X_next, y, info)
    contract, so a loop driver can swap this in for the real leaf-index
    condition without changing anything else about the run.

    leaf_ids : optional (n, ntree) array from an already-fitted real forest
        (leaf_index_features.fit_leaf_forest's second return value), to
        measure matched granularity from without paying for a second,
        redundant forest fit when a loop already fit the real condition
        this generation. If None, fits one internally via `config` --
        X and y are used ONLY to measure this real forest's granularity;
        the resulting partition ids never depend on which samples produced
        which leaf, only on how many cells of what sizes existed.
    mode="replace"    : X_next = Z only (primary condition).
    mode="accumulate" : X_next = [X, Z] (E4-style comparison). Under
        "onehot" this returns a sparse matrix; under "ordinal" it stays
        dense.
    encoding="onehot" (default) or "ordinal" -- match whichever encoding
        the real leaf-index condition is being compared against in this run.

    Returns
    -------
    X_next, y, info : info holds `cell_sizes` (needed by
        transform_random_partition for held-out data), `encoder` (None for
        "ordinal"), and `encoding`.
    """
    if mode not in ("replace", "accumulate"):
        raise ValueError(f"mode must be 'replace' or 'accumulate', got {mode!r}")
    if encoding not in VALID_ENCODINGS:
        raise ValueError(f"encoding must be one of {VALID_ENCODINGS}, got {encoding!r}")

    if leaf_ids is None:
        _, leaf_ids = fit_leaf_forest(X, y, config, random_state)
    cell_sizes = cell_sizes_from_leaf_ids(leaf_ids)

    partition_ids = random_partition_ids(len(y), cell_sizes, random_state)
    Z, encoder = leaf_features(partition_ids, encoding)

    if mode == "replace":
        X_next = Z
    elif encoding == "ordinal":
        X_next = np.hstack([X, Z])
    else:
        X_next = sp.hstack([sp.csr_matrix(X), Z]).tocsr()

    info = dict(cell_sizes=cell_sizes, encoder=encoder, encoding=encoding)
    return X_next, y, info


def transform_random_partition(X_query, info, random_state, mode="replace"):
    """Score-time counterpart to recycle_random_partition, for held-out data
    (e.g. the fixed real test set). There is nothing to transfer from the
    training partition -- see module docstring -- so this draws a FRESH,
    independent partition of the query rows with the SAME per-tree
    cell-size multiset (`info["cell_sizes"]`), encoded the same way
    (`info["encoding"]`, reusing `info["encoder"]` for "onehot" so the
    one-hot column layout matches what the generation's classifier was
    trained on).

    `random_state` should be independent of the training partition's seed
    (e.g. derived via experiments._common.child_seeds) -- reusing the same
    seed would not "align" query cells with training cells in any
    meaningful sense (there is a different number of query rows to begin
    with), it would just be an arbitrary coincidence, not a real one.
    """
    if mode not in ("replace", "accumulate"):
        raise ValueError(f"mode must be 'replace' or 'accumulate', got {mode!r}")

    partition_ids = random_partition_ids(X_query.shape[0], info["cell_sizes"], random_state)
    Z, _ = leaf_features(partition_ids, info["encoding"], info["encoder"])

    if mode == "replace":
        return Z
    if info["encoding"] == "ordinal":
        return np.hstack([X_query, Z])
    return sp.hstack([sp.csr_matrix(X_query), Z]).tocsr()


if __name__ == "__main__":
    # python -m experiments.random_partition_features -- relative imports
    # (._common, .leaf_index_features) need package context.
    from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

    from datasets.generate_tier_a_dataset import make_geometry, make_splits
    from .metrics import mutual_information_x_leaf

    geom = make_geometry()
    train, test, _ = make_splits(geom)

    config = LeafIndexConfig(ntree=50, nodesize=50)  # small/fast, demo only
    real_forest, real_leaf_ids = fit_leaf_forest(train.X, train.y, config, random_state=0)
    print(f"real forest: {config.ntree} trees, "
          f"leaves/tree range {[len(np.unique(real_leaf_ids[:, t])) for t in range(3)]}...")

    for encoding in VALID_ENCODINGS:
        train_seed, test_seed = child_seeds(0, 2)
        X_next, y_next, info = recycle_random_partition(
            train.X, train.y, config, random_state=train_seed,
            mode="replace", encoding=encoding, leaf_ids=real_leaf_ids,
        )
        n_cells_match = all(
            len(info["cell_sizes"][t]) == len(np.unique(real_leaf_ids[:, t]))
            for t in range(config.ntree)
        )
        print(f"\n{encoding}: X_next shape={X_next.shape}, "
              f"cell counts match real forest: {n_cells_match}")

        gen1 = RandomForestClassifier(
            n_estimators=config.ntree, min_samples_leaf=config.nodesize,
            max_features=config.resolve_mtry(X_next.shape[1]), n_jobs=config.n_jobs,
            random_state=0,
        ).fit(X_next, y_next)

        X_test_next = transform_random_partition(test.X, info, test_seed, mode="replace")
        acc = gen1.score(X_test_next, test.y)
        majority_rate = np.bincount(test.y).max() / len(test.y)
        print(f"  gen-1 test acc={acc:.4f}  (majority-class rate={majority_rate:.4f} "
              f"-- expect acc close to this, since partition carries no real signal)")

    partition_ids = random_partition_ids(len(train.y), cell_sizes_from_leaf_ids(real_leaf_ids), 0)
    mi_random, _ = mutual_information_x_leaf(train.X, partition_ids)
    mi_real, _ = mutual_information_x_leaf(train.X, real_leaf_ids)
    print(f"\nI(X;L) real leaves:      {mi_real:.4f}")
    print(f"I(X;L) random partition: {mi_random:.4f}  (NOT ~0 -- mutual_info_score has a "
          f"well-known finite-sample upward bias when category counts are large relative to "
          f"n (two truly independent variables with 10x150 categories at n=10000 already "
          f"score ~0.07 per feature); what matters is that it sits near that estimator floor, "
          f"far below the real leaves' MI, not that it is literally zero)")
