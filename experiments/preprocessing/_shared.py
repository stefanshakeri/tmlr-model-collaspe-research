"""
Shared helpers for the Tier B preprocessing modules (covertype.py,
income.py). Factored out because both need the exact same two operations --
stratified subsampling of existing rows, and correlation-based redundant-
continuous-feature pruning -- and duplicating either would risk the two
datasets silently drifting to different definitions of "rare" or "redundant"
for no principled reason. (experiments/_common.py holds the more general
RF-seeding/mtry helpers used outside preprocessing too; this module is
preprocessing-specific.)
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np


def stratified_indices(y, n, rng):
    """Proportional-allocation stratified sample of `n` indices into `y`,
    without replacement within each stratum. `y` can be any discrete
    grouping variable -- a target label, a demographic category, or a
    combined "label|group" key (Income's make_splits joins income and race
    this way, to protect small racial subgroups from uneven train/test
    placement, not just the income split) -- this only needs it to be
    categorical.
    """
    labels, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    alloc = rng.multinomial(n, counts / counts.sum())
    idx_parts = []
    for lbl_i, take in enumerate(alloc):
        pool = np.flatnonzero(inverse == lbl_i)
        idx_parts.append(rng.choice(pool, size=min(take, len(pool)), replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return idx


@dataclass
class CorrelationPruning:
    """A fixed, population-level decision about which continuous features
    are redundant -- computed ONCE from the full population and reused
    across every subsample/split seed. Recomputing this per-subsample would
    let the pruned feature set vary by sample_seed, which breaks fi_drift's
    requirement (experiments/metrics.py) that a generation-k and a
    generation-0 importance vector index the same columns, and would
    silently change which continuous columns W2 runs over from run to run.
    """
    threshold: float
    keep_idx: np.ndarray  # indices into the (unpruned) continuous block, kept
    dropped: list          # [(dropped_name, kept_partner_name, r), ...], most-correlated first

    def to_dict(self):
        return {"threshold": self.threshold, "keep_idx": self.keep_idx.tolist(),
                "dropped": self.dropped}

    def config_id(self):
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:10]


def correlation_matrix(continuous_X):
    """Pearson correlation matrix of an already-sliced continuous-feature
    matrix. Each dataset module slices its own continuous block first (see
    that module's continuous_features()) -- correlating one-hot categorical
    columns would mostly recover their own within-block mutual exclusivity,
    not genuine redundancy, for either dataset here."""
    return np.corrcoef(continuous_X, rowvar=False)


def highly_correlated_pairs(corr, threshold=0.9):
    """(i, j, r) triples of column indices with |corr[i, j]| >= threshold,
    i < j, most-correlated first. threshold=0.9 is the conventional
    multicollinearity cutoff; lower it to flag more aggressively."""
    p = corr.shape[0]
    pairs = [
        (i, j, float(corr[i, j]))
        for i in range(p) for j in range(i + 1, p)
        if abs(corr[i, j]) >= threshold
    ]
    pairs.sort(key=lambda t: -abs(t[2]))
    return pairs


def fit_correlation_pruning(continuous_X, feature_names, threshold=0.9):
    """Decide which continuous features to drop as redundant, from an
    already-sliced (population-level -- see CorrelationPruning's docstring)
    continuous feature matrix.

    Greedy findCorrelation-style heuristic: for each offending pair (|r| >=
    threshold, strongest first), drop whichever of the two features has the
    larger mean |correlation| against every OTHER continuous feature (not
    just its partner in this pair) -- a feature central to several redundant
    pairs is removed before one only weakly duplicated once. A feature
    already dropped by a stronger pair is left alone when a later, weaker
    pair involving it is considered.
    """
    corr = correlation_matrix(continuous_X)
    p = corr.shape[0]
    mean_abs_corr = (np.abs(corr).sum(axis=1) - 1) / (p - 1)  # exclude self

    dropped_idx = set()
    dropped_log = []
    for i, j, r in highly_correlated_pairs(corr, threshold):
        if i in dropped_idx or j in dropped_idx:
            continue
        drop, keep = (i, j) if mean_abs_corr[i] >= mean_abs_corr[j] else (j, i)
        dropped_idx.add(drop)
        dropped_log.append((feature_names[drop], feature_names[keep], r))

    keep_idx = np.array([k for k in range(p) if k not in dropped_idx])
    return CorrelationPruning(threshold=threshold, keep_idx=keep_idx, dropped=dropped_log)


def apply_pruning(X, pruning: CorrelationPruning, n_continuous):
    """Drop the redundant continuous columns `pruning` identified from the
    first `n_continuous` columns of X, leaving every column after that
    (one-hot blocks) untouched and appended unchanged after the surviving
    continuous columns. Apply the SAME `pruning` object (fit once on the
    population) to every split -- never refit per split, for the reason
    CorrelationPruning's docstring explains.
    """
    cont = X[:, :n_continuous][:, pruning.keep_idx]
    rest = X[:, n_continuous:]
    return np.hstack([cont, rest])
