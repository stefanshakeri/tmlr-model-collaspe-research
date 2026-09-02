"""
Per-generation and trajectory-level collapse metrics, one function (or small
function group) per row of local-docs/metrics.csv. Log column names in that
CSV are used verbatim as the keys `compute_generation_metrics` returns, so a
results row can be written straight to CSV/parquet without a renaming step.

Every function here is a standalone primitive an experiment notebook can call
directly -- `compute_generation_metrics` is a convenience wrapper, not the
only entry point. Metrics fall into two groups:

  Per-generation (metrics.csv #1-12): computed once per (generation, seed,
  condition) from that generation's fitted model and the fixed real test set.

  Trajectory-level (metrics.csv #13-14): computed AFTER a full run, from the
  sequence of one per-generation metric's values across generations (#13,
  one trajectory) or across seeds (#14, many trajectories). Suresh et al.'s
  point (experimental_design.md 3.2) is that these, not the per-generation
  values alone, are what make a rate-of-collapse claim rather than an
  endpoint anecdote.

A design note that applies across several metrics here: several of them
(n_leaves_occupied, leaf_entropy, mi_x_leaf, w2_feature) need per-sample,
per-tree LEAF ASSIGNMENTS -- `clf.apply(X)`, an (n, n_trees) int array --
which is exactly what synthetic_features.py and leaf_index_features.py
already compute internally. This module takes leaf ids (or a fitted `clf`)
as plain arguments rather than importing those modules, so it stays usable
regardless of which recycling scheme, or no recycling at all, produced the
generation being scored.
"""

from typing import Optional

import numpy as np
import ot  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, mutual_info_score  # type: ignore[import-untyped]
from sklearn.preprocessing import KBinsDiscretizer  # type: ignore[import-untyped]
from scipy.stats import spearmanr  # type: ignore[import-untyped]

GENERATION_METRIC_COLUMNS = (
    "acc_test", "n_leaves_occupied", "recall_minority", "recall_rare",
    "acc_tail", "acc_tail_eval", "pred_prob_var", "tree_disagreement",
    "w2_feature", "leaf_entropy", "mi_x_leaf", "fi_drift",
)


# ---------------------------------------------------------------------------
# #1 acc_test / #6 acc_tail_eval -- both are exactly this, on different sets.
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    """Plain accuracy. Used directly for acc_test (fixed real test set) and
    acc_tail_eval (Tier A's tail-enriched evaluation set) -- they differ only
    in which set is passed in, not in how they're computed."""
    return float(accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# #3 recall_minority
# ---------------------------------------------------------------------------

def recall_minority(y_true, y_pred, minority_class=None):
    """Recall on the minority class of the (fixed, real) test set.
    minority_class is auto-detected from y_true's class counts if omitted --
    safe to do every call since y_true is the never-regenerated test set and
    its class balance is constant across generations."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if minority_class is None:
        counts = np.bincount(y_true)
        minority_class = int(np.argmin(np.where(counts > 0, counts, np.inf)))
    mask = y_true == minority_class
    if not mask.any():
        raise ValueError(f"no samples of class {minority_class} in y_true")
    return float((y_pred[mask] == minority_class).mean())


# ---------------------------------------------------------------------------
# #4 recall_rare / #5 acc_tail -- both masked accuracy, different masks.
# ---------------------------------------------------------------------------

def masked_accuracy(y_true, y_pred, mask):
    """Accuracy restricted to `mask`. #4 (recall_rare) passes
    TierASplit.in_rare, #5 (acc_tail) passes TierASplit.in_tail -- the two
    independent tail definitions from datasets/generate_tier_a_dataset.py.
    Returns NaN (not an error) on an empty mask, since a mask legitimately
    being empty at some late generation is itself the finding (total tail
    loss), not a bug."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float((y_pred[mask] == y_true[mask]).mean())


# ---------------------------------------------------------------------------
# #2 n_leaves_occupied / #10 leaf_entropy -- both from per-tree leaf counts.
# ---------------------------------------------------------------------------

def _leaf_cell_counts(leaf_ids):
    """Per-tree occupancy counts: for each tree, how many samples landed in
    each leaf it actually produced. A (tree, leaf) pair is one "cell" --
    leaf 7 in tree 3 and leaf 7 in tree 9 are unrelated cells, since leaf ids
    are only meaningful within their own tree. Shared by n_leaves_occupied
    and leaf_entropy so both use the same definition of "a cell"."""
    return [np.unique(leaf_ids[:, t], return_counts=True)[1]
            for t in range(leaf_ids.shape[1])]


def n_leaves_occupied(leaf_ids):
    """Total occupied (tree, leaf) cells across the whole forest -- the
    classification analog of Suresh et al.'s uniq_k (surviving distinct
    symbols after k generations)."""
    return int(sum(len(c) for c in _leaf_cell_counts(leaf_ids)))


def n_leaves_occupied_from_encoder(encoder):
    """Same quantity, read directly off a fitted OneHotEncoder's categories_
    (leaf_index_features.py's onehot_leaf_features) instead of recomputing
    from leaf ids -- the "free from OneHotEncoder.categories_" shortcut
    metrics.csv notes for this metric."""
    return int(sum(len(c) for c in encoder.categories_))


def leaf_entropy(leaf_ids):
    """Shannon entropy (nats) of the pooled leaf-occupancy distribution --
    every (sample, tree) assignment treated as one draw from the same
    categorical variable over (tree, leaf) cells. The discrete-space
    substitute for W2 that needs no ground metric (see wasserstein2_feature
    below for why W2 on raw leaf ids specifically is invalid)."""
    counts = np.concatenate(_leaf_cell_counts(leaf_ids))
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# #7 pred_prob_var
# ---------------------------------------------------------------------------

def prediction_probability_variance(proba):
    """Mean, across classes, of each class's predicted-probability variance
    across the test set. The classification analog of Shumailov's sigma -> 0:
    a model collapsing toward degenerate, near-constant predictions has this
    shrink toward zero regardless of whether accuracy has visibly moved yet."""
    proba = np.asarray(proba)
    return float(np.mean(np.var(proba, axis=0)))


# ---------------------------------------------------------------------------
# #8 tree_disagreement
# ---------------------------------------------------------------------------

def per_tree_predictions(clf, X):
    """(n, n_trees) hard predictions, one column per tree in the forest."""
    return np.column_stack([tree.predict(X) for tree in clf.estimators_])


def tree_disagreement(clf, X):
    """Mean pairwise disagreement rate between individual trees' hard
    predictions, averaged over the test set. For each sample, if c_j trees
    vote class j, the number of agreeing tree-pairs is sum_j C(c_j, 2); this
    is the closed-form equivalent of averaging disagreement over every
    O(n_trees^2) tree pair without actually enumerating them. Alemohammad's
    diversity/recall axis: distinct from accuracy, since a forest can stay
    accurate on average while its trees increasingly agree with each other
    (diversity collapse) rather than with the ground truth."""
    n_trees = len(clf.estimators_)
    if n_trees < 2:
        raise ValueError("tree_disagreement needs >= 2 trees")
    preds = per_tree_predictions(clf, X)
    class_idx = np.searchsorted(clf.classes_, preds)
    n_samples = preds.shape[0]
    n_classes = len(clf.classes_)
    counts = np.zeros((n_samples, n_classes))
    for c in range(n_classes):
        counts[:, c] = (class_idx == c).sum(axis=1)
    total_pairs = n_trees * (n_trees - 1) / 2
    agree_pairs = (counts * (counts - 1) / 2).sum(axis=1)
    disagreement = 1.0 - agree_pairs / total_pairs
    return float(disagreement.mean())


# ---------------------------------------------------------------------------
# #9 w2_feature -- NEVER on raw leaf ids. See caveat below.
# ---------------------------------------------------------------------------

def leaf_centroid_reconstruction(leaf_ids_query, leaf_ids_train, feature_train):
    """Map each query row to the average, across trees, of its leaf's
    centroid in `feature_train` -- computed from training rows aligned with
    leaf_ids_train, NOT necessarily the same feature space clf.apply() used
    to produce the leaf ids. That decoupling is deliberate: it lets you ask
    "what does this generation's tree partition imply about the ORIGINAL
    continuous features" even when the model itself was fit on a recycled
    representation (OOB predictions or leaf indices), by passing Tier A's
    true continuous features as `feature_train` while leaf_ids_train/query
    come from clf.apply() on whatever `X_train_model` actually was.

    A leaf id in leaf_ids_query that leaf_ids_train never produced is a
    contract violation, not a modeling edge case: leaf_ids_train must come
    from the exact data the tree was fit on, and a fitted tree cannot
    contain a leaf that no training sample created. Raise loudly rather than
    silently falling back to a global mean.
    """
    n_trees = leaf_ids_train.shape[1]
    recon = np.zeros((leaf_ids_query.shape[0], feature_train.shape[1]))
    for t in range(n_trees):
        leaves_t, inv_train = np.unique(leaf_ids_train[:, t], return_inverse=True)
        sums = np.zeros((len(leaves_t), feature_train.shape[1]))
        np.add.at(sums, inv_train, feature_train)
        counts = np.bincount(inv_train, minlength=len(leaves_t))
        centroids = sums / counts[:, None]

        idx = np.searchsorted(leaves_t, leaf_ids_query[:, t])
        idx_clipped = np.clip(idx, 0, len(leaves_t) - 1)
        if not np.array_equal(leaves_t[idx_clipped], leaf_ids_query[:, t]):
            raise ValueError(
                f"tree {t}: a query leaf id was never produced by "
                f"leaf_ids_train. leaf_ids_train must come from the same "
                f"data the forest was fit on."
            )
        recon += centroids[idx]
    recon /= n_trees
    return recon


def wasserstein2_feature(X_true, X_recon, max_samples=1000, random_state=0):
    """W2 between the true continuous feature distribution and the
    distribution reconstructed from leaf centroids (leaf_centroid_
    reconstruction above), via POT's exact discrete optimal transport.

    NEVER call this on raw leaf indices directly -- they are nominal, with
    no meaningful ground metric, so any W2 computed on them is an artifact
    of arbitrary index numbering (experimental_design.md 3.1). This function
    only accepts continuous feature matrices; leaf_centroid_reconstruction
    exists precisely to turn a discrete leaf assignment into one.

    Exact OT is O(n^3); `max_samples` subsamples both sides (without
    replacement, same `random_state` for the two draws so repeated calls at
    the same seed are comparable) to keep this tractable. Also unreliable
    above d~20 (sample complexity scales as n^(-1/d)) -- caller's problem to
    keep d in range, not this function's to enforce.
    """
    X_true = np.asarray(X_true)
    X_recon = np.asarray(X_recon)
    rng = np.random.default_rng(random_state)
    n = min(len(X_true), len(X_recon), max_samples)
    A = X_true[rng.choice(len(X_true), n, replace=False)]
    B = X_recon[rng.choice(len(X_recon), n, replace=False)]

    M = ot.dist(A, B, metric="sqeuclidean")
    a = np.full(n, 1.0 / n)
    b = np.full(n, 1.0 / n)
    w2_squared = ot.emd2(a, b, M)
    return float(np.sqrt(w2_squared))


# ---------------------------------------------------------------------------
# #11 mi_x_leaf
# ---------------------------------------------------------------------------

def mutual_information_x_leaf(X, leaf_ids, n_bins=10):
    """I(X; L), the KEY CONTROL METRIC for E2 (experimental_design.md 4, E2):
    how much information the current partition retains about the original
    features, to be plotted against degradation as E2 sweeps partition
    granularity.

    Computed as the sum, over original features, of each feature's mutual
    information with the leaf assignment (averaged over trees), after
    quantile-binning the feature. Binning makes both sides of the
    contingency table discrete, so sklearn's mutual_info_score is then EXACT
    for the binned proxy -- no neural or kNN MI estimator needed, per
    metrics.csv's own note. The approximation is entirely in the binning and
    in summing single-feature MIs as a tractable proxy for the true joint
    I(X_1..X_p; L); an exact p-dimensional joint contingency table is
    combinatorially infeasible for p >= ~4, so this is a standard
    substitute, not the literal joint MI.

    Returns
    -------
    total       : float, sum over features of the tree-averaged per-feature MI.
    per_feature : (p,) array the total is built from, for diagnosing which
                  features are driving it.
    """
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    n_trees = leaf_ids.shape[1]
    binned = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="quantile"
    ).fit_transform(X).astype(int)

    per_feature = np.zeros(p)
    for j in range(p):
        per_feature[j] = np.mean([
            mutual_info_score(binned[:, j], leaf_ids[:, t])
            for t in range(n_trees)
        ])
    return float(per_feature.sum()), per_feature


# ---------------------------------------------------------------------------
# #12 fi_drift
# ---------------------------------------------------------------------------

def feature_importance_drift(importances_k, importances_0, method="l1"):
    """Distance between generation-k and generation-0 feature_importances_.
    Only meaningful when the feature space means the same thing across
    generations -- "accumulate" mode, or a fixed synthetic-feature layout
    (SRF's stacked nodesize blocks). Under "replace" mode with leaf-index or
    OOB-prediction recycling, the feature space itself changes identity
    every generation, so this comparison is not well-defined; don't call it
    there, rather than silently comparing unrelated columns.

    method="l1" (default): sum |a - b|.
    method="spearman": 1 - Spearman rank correlation (0 = identical rank
    order of feature importances, 2 = perfectly reversed).
    """
    a = np.asarray(importances_k)
    b = np.asarray(importances_0)
    if len(a) != len(b):
        raise ValueError(
            f"importance vectors must share a feature space to compare "
            f"drift, got lengths {len(a)} vs {len(b)}"
        )
    if method == "l1":
        return float(np.abs(a - b).sum())
    if method == "spearman":
        rho, _ = spearmanr(a, b)
        return float(1.0 - rho)
    raise ValueError(f"method must be 'l1' or 'spearman', got {method!r}")


# ---------------------------------------------------------------------------
# #13 gens_to_thresh / #14 p_collapse_k -- trajectory-level, computed post-hoc.
# ---------------------------------------------------------------------------

def generations_to_threshold(metric_trajectory, frac=0.5, baseline=None):
    """First generation g (>=1) where metric_trajectory[g] < frac * baseline,
    or None if the trajectory never crosses it. baseline defaults to
    metric_trajectory[0] (this trajectory's own generation-0 value).

    `metric_trajectory` is ONE seed's sequence of a single per-generation
    metric across generations 0..G -- e.g. one row of a (seeds, generations)
    matrix for recall_minority. Suresh et al.'s point (experimental_design.md
    3.2): the RATE of collapse is more informative than whether it eventually
    happens, and this is cheap to compute from data you're already logging.
    """
    arr = np.asarray(metric_trajectory, dtype=float)
    if baseline is None:
        baseline = arr[0]
    below = np.flatnonzero(arr < frac * baseline)
    return int(below[0]) if len(below) else None


def probability_collapse_by_generation(trajectories, frac=0.5, baseline=None):
    """p[k] = fraction of seeds whose trajectory has crossed the
    generations_to_threshold criterion by generation k (monotone
    nondecreasing in k). `trajectories` is (n_seeds, n_generations) -- many
    seeds' sequences of the SAME per-generation metric. Xu et al.'s
    probabilistic framing (experimental_design.md 3.3): most runs may
    collapse to near-zero while a few diverge, so this -- not a mean
    trajectory -- is what makes multi-seed reporting rigorous rather than
    decorative.

    `baseline`, if given, is shared across all seeds (e.g. a fixed
    calibration target); if None, each seed uses its own generation-0 value.
    """
    trajectories = np.asarray(trajectories, dtype=float)
    n_seeds, n_gens = trajectories.shape
    collapsed_at = np.full(n_seeds, np.inf)
    for s in range(n_seeds):
        g = generations_to_threshold(trajectories[s], frac, baseline)
        if g is not None:
            collapsed_at[s] = g
    return np.array([(collapsed_at <= k).mean() for k in range(n_gens)])


# ---------------------------------------------------------------------------
# Convenience aggregator: everything computable this generation, in one call.
# ---------------------------------------------------------------------------

def compute_generation_metrics(
    clf,
    X_train_model,
    test_X, test_y,
    tail_eval_X=None, tail_eval_y=None,
    in_rare=None, in_tail=None,
    minority_class=None,
    baseline_importances: Optional[np.ndarray] = None,
    fi_method="l1",
    true_feature_train=None, true_feature_test=None,
    mi_n_bins=10,
    w2_max_samples=1000, w2_random_state=0,
):
    """Compute every per-generation metric (#1-12) whose required inputs are
    supplied, keyed by metrics.csv's Log column names -- ready to append as
    one row to a results table. A metric whose inputs are omitted (None) is
    simply ABSENT from the returned dict, not set to NaN, since "not
    applicable this condition" (e.g. no rare-subpopulation mask on Tier B)
    differs from "computed and undefined" (an empty mask, which
    masked_accuracy does report as NaN).

    clf            : this generation's fitted classifier.
    X_train_model  : training features exactly as clf was fit on (whatever
                     representation the current recycling condition
                     produced).
    test_X, test_y : the fixed, never-regenerated real held-out set, in the
                     SAME representation as X_train_model.
    tail_eval_X/_y : Tier A's tail-enriched evaluation set (Tier A only).
    in_rare, in_tail : boolean masks over test_y (TierASplit.in_rare/.in_tail).
    minority_class : test set's minority class; auto-detected if omitted.
    baseline_importances : generation-0 clf.feature_importances_, for
                     fi_drift -- see that function's docstring for when this
                     comparison is (and isn't) meaningful.
    true_feature_train/_test : ORIGINAL continuous features, aligned
                     row-for-row with X_train_model/test_X, for w2_feature
                     and mi_x_leaf. Tier A only.
    """
    out = {}
    y_pred = clf.predict(test_X)
    out["acc_test"] = accuracy(test_y, y_pred)
    out["recall_minority"] = recall_minority(test_y, y_pred, minority_class)

    if in_rare is not None:
        out["recall_rare"] = masked_accuracy(test_y, y_pred, in_rare)
    if in_tail is not None:
        out["acc_tail"] = masked_accuracy(test_y, y_pred, in_tail)
    if tail_eval_X is not None:
        out["acc_tail_eval"] = accuracy(tail_eval_y, clf.predict(tail_eval_X))

    proba = clf.predict_proba(test_X)
    out["pred_prob_var"] = prediction_probability_variance(proba)
    out["tree_disagreement"] = tree_disagreement(clf, test_X)

    leaf_ids_train = clf.apply(X_train_model)
    out["n_leaves_occupied"] = n_leaves_occupied(leaf_ids_train)
    out["leaf_entropy"] = leaf_entropy(leaf_ids_train)

    if true_feature_train is not None:
        out["mi_x_leaf"], _ = mutual_information_x_leaf(
            true_feature_train, leaf_ids_train, mi_n_bins
        )
        if true_feature_test is not None:
            leaf_ids_test = clf.apply(test_X)
            recon = leaf_centroid_reconstruction(
                leaf_ids_test, leaf_ids_train, true_feature_train
            )
            out["w2_feature"] = wasserstein2_feature(
                true_feature_test, recon, w2_max_samples, w2_random_state
            )

    if baseline_importances is not None:
        out["fi_drift"] = feature_importance_drift(
            clf.feature_importances_, baseline_importances, fi_method
        )

    return out


if __name__ == "__main__":
    # python -m experiments.metrics -- see experiments/synthetic_features.py
    # for why (relative imports need package context).
    from datasets.generate_tier_a_dataset import make_geometry, make_splits
    from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

    geom = make_geometry()
    train, test, tail_eval = make_splits(geom)
    clf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0,
        oob_score=False,
    ).fit(train.X, train.y)

    gen0 = compute_generation_metrics(
        clf, train.X, test.X, test.y,
        tail_eval_X=tail_eval.X, tail_eval_y=tail_eval.y,
        in_rare=test.in_rare, in_tail=test.in_tail,
        baseline_importances=clf.feature_importances_,  # trivially 0 drift at gen 0
        true_feature_train=train.X, true_feature_test=test.X,
    )
    print(f"generation-0 metrics (n={len(train.y)}, p={train.X.shape[1]}):\n")
    for col in GENERATION_METRIC_COLUMNS:
        if col in gen0:
            print(f"  {col:20s} {gen0[col]:.4f}")

    # Trajectory-level demo on a synthetic recall_minority-like series.
    trajectories = np.array([
        [1.00, 0.95, 0.80, 0.55, 0.30, 0.10],
        [1.00, 0.98, 0.93, 0.85, 0.70, 0.60],
        [1.00, 0.60, 0.20, 0.05, 0.01, 0.00],
    ])
    print(f"\ngenerations_to_threshold (frac=0.5) per seed: "
          f"{[generations_to_threshold(t) for t in trajectories]}")
    print(f"p_collapse_k (frac=0.5): {probability_collapse_by_generation(trajectories)}")
