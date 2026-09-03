"""
Tier B preprocessing: Covertype, per local-docs/experimental_design.md's
Dataset selection > Tier B:

    "Covertype, subsampled to ~50k rows (full set is 581k -- subsample it).
    Its value is that it has naturally rare classes (two classes at roughly
    1% prevalence), which makes 'how many classes has the model forgotten by
    generation k' directly measurable -- the classification analog of Suresh
    et al.'s surviving-symbol count."

and local-docs/metrics.csv, whose Tier B column marks several metrics
differently than Tier A:

  - recall_rare : "Partial -- Tier B needs a rare-class proxy." Tier A's
    in_rare is membership in a deliberately-placed satellite cluster, which
    real data doesn't have. Here the proxy is literal class-rarity: classes
    below `rare_threshold` population prevalence (UCI's Covertype has two
    such classes, Cottonwood/Willow at ~0.5% and Aspen at ~1.6%). Exposed as
    TierBSplit.in_rare so it plugs into experiments.metrics.masked_accuracy
    and compute_generation_metrics(in_rare=...) exactly like Tier A's field
    of the same name.
  - acc_tail, acc_tail_eval : "No." There is no ground-truth density tail on
    real data (Tier A's Mahalanobis definition needs a known class-conditional
    distribution), so this module does not produce in_tail or a tail-eval
    split at all -- not "produces one that's always empty."
  - w2_feature : "Limited." Only the 10 genuinely continuous columns
    (Elevation..Horizontal_Distance_To_Fire_Points) are meaningful under a
    Euclidean ground metric; the other 44 columns are binary one-hot
    indicators (Wilderness_Area x4, Soil_Type x40). continuous_features()
    and standardize_continuous() below exist specifically to hand
    experiments.metrics.wasserstein2_feature the right, scale-comparable
    subset -- never the full 54-column matrix.

Structure/sample-seed split mirrors datasets/generate_tier_a_dataset.py:
CovertypeGeometry (population class proportions, the rare-class proxy set)
and CorrelationPruning (redundant-continuous-feature set) are both fixed
ONCE from the full 581,012-row population and reused across every subsample
seed -- recomputing either from a 50k subsample instead would let the
definitions themselves drift with sampling noise across seeds and
generations, exactly the failure Tier A's Geometry.tail_threshold is built
to avoid. For pruning specifically, letting it vary by seed would also break
fi_drift's requirement (experiments/metrics.py) that a generation-k and a
generation-0 importance vector index the same columns.

The raw fetch + local cache (fetch_covertype_raw) lives in
datasets/covertype_dataset.py, not here: this module only ever consumes
(X, y) arrays that module hands back, and knows nothing about UCI, network
I/O, or the on-disk cache format. That split is deliberate -- datasets/ is
"what raw data did we get", experiments/preprocessing/ is "what does an
experiment need done to it" -- so a change to subsampling strategy or the
rare-class threshold never touches the fetch/cache path and vice versa.
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .._common import child_seeds
from datasets.covertype_dataset import (
    fetch_covertype_raw, N_CONTINUOUS, DEFAULT_CACHE_DIR,
)


@dataclass
class TierBSplit:
    """Real-data analog of datasets/generate_tier_a_dataset.py's TierASplit.
    No in_tail/mahalanobis fields -- Tier B has no ground-truth density tail
    (metrics.csv: acc_tail is "No" for Tier B); in_rare is the rare-class
    proxy described in the module docstring, not a satellite subpopulation.
    """
    X: np.ndarray       # (n, 54) features: 10 continuous + 44 binary one-hot
    y: np.ndarray        # (n,) 0-indexed class labels (UCI's Cover_Type - 1)
    in_rare: np.ndarray  # (n,) bool -- membership in a CovertypeGeometry.rare_classes label

    def summary(self):
        return {
            "n": len(self.y),
            "d": self.X.shape[1],
            "class_balance": np.bincount(
                self.y, minlength=int(self.y.max()) + 1
            ).astype(float) / len(self.y),
            "rare_frac": float(self.in_rare.mean()),
        }


@dataclass
class CovertypeGeometry:
    """Fixed structure: population class proportions and the resulting
    rare-class proxy set, computed once from the full population. Build
    once, reuse across every subsample/split seed -- see module docstring.
    """
    class_labels: np.ndarray   # sorted unique 0-indexed labels, 0..6
    class_props: np.ndarray    # population-level proportion per label
    rare_classes: np.ndarray   # labels with population proportion < rare_threshold
    rare_threshold: float
    n_population: int

    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return out

    def config_id(self):
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:10]


def make_geometry(rare_threshold=0.02, cache_dir=DEFAULT_CACHE_DIR):
    """Population class proportions and the rare-class proxy set (see module
    docstring). Default rare_threshold=0.02 isolates exactly the two classes
    experimental_design.md refers to (Cottonwood/Willow ~0.5% and Aspen
    ~1.6% of the population) without also catching Douglas-fir (~3%).
    """
    _, y_full = fetch_covertype_raw(cache_dir)
    labels, counts = np.unique(y_full, return_counts=True)
    props = counts / counts.sum()
    rare = labels[props < rare_threshold]
    return CovertypeGeometry(
        class_labels=labels, class_props=props, rare_classes=rare,
        rare_threshold=rare_threshold, n_population=len(y_full),
    )


def _stratified_indices(y, n, rng):
    """Proportional-allocation stratified sample of `n` indices into `y`,
    without replacement within each class. The real-data analog of
    datasets/generate_tier_a_dataset.py's multinomial class allocation in
    `sample()`, adapted to sample INDICES into existing rows rather than
    draw new points -- preserving, not distorting, whatever rare classes
    are present in `y`, which is the entire reason to use Covertype."""
    labels, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    alloc = rng.multinomial(n, counts / counts.sum())
    idx_parts = []
    for lbl_i, take in enumerate(alloc):
        pool = np.flatnonzero(inverse == lbl_i)
        idx_parts.append(rng.choice(pool, size=min(take, len(pool)), replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return idx


def make_splits(geom, n_subsample=50_000, test_frac=0.2, sample_seed=0,
                 cache_dir=DEFAULT_CACHE_DIR):
    """Stratified ~n_subsample-row draw from the full population (proportional
    allocation, so the subsample's class balance matches the population's --
    experimental_design.md 5: "There is no scientific reason to use all of
    it," but subsampling naively i.i.d. at n=50k would still roughly preserve
    balance on its own; stratifying removes even that residual seed-to-seed
    variance in how many rare-class rows land in the subsample), then a
    further stratified train/test split of the subsample.

    The test split is the fixed, never-regenerated real held-out set: draw
    it once per sample_seed and never call make_splits again for the same
    run (identical discipline to Tier A's make_splits).
    """
    X_full, y_full = fetch_covertype_raw(cache_dir)
    subsample_seed, split_seed = child_seeds(sample_seed, 2)

    sub_idx = _stratified_indices(y_full, n_subsample, np.random.default_rng(subsample_seed))
    X_sub, y_sub = X_full[sub_idx], y_full[sub_idx]

    n_test = int(round(len(y_sub) * test_frac))
    test_idx = _stratified_indices(y_sub, n_test, np.random.default_rng(split_seed))
    test_mask = np.zeros(len(y_sub), dtype=bool)
    test_mask[test_idx] = True

    def _split(mask):
        return TierBSplit(
            X=X_sub[mask], y=y_sub[mask],
            in_rare=np.isin(y_sub[mask], geom.rare_classes),
        )

    return _split(~test_mask), _split(test_mask)


def continuous_features(X, n_continuous=N_CONTINUOUS):
    """The first `n_continuous` columns -- the only ones
    experiments.metrics.wasserstein2_feature should be run on for Tier B.
    See module docstring's w2_feature entry for why. Defaults to all 10 raw
    continuous columns; pass len(pruning.keep_idx) after apply_pruning() has
    dropped some of them, so this still slices the right width."""
    return X[:, :n_continuous]


def standardize_continuous(train_X, *other_X, n_continuous=N_CONTINUOUS):
    """Z-score the continuous columns using TRAIN statistics only, applied to
    train_X and every array in other_X (e.g. the fixed test split). W2's
    Euclidean ground metric would otherwise be dominated by Elevation and the
    distance features (raw scale: thousands) over Aspect/Hillshade (raw
    scale: 0-360) -- an artifact of units, not signal. Never refit on
    anything but the fixed train split, for the same leakage reason Tier A's
    test set is never regenerated.
    """
    train_cont = continuous_features(train_X, n_continuous)
    mean = train_cont.mean(axis=0)
    std = train_cont.std(axis=0)
    std[std == 0] = 1.0  # a constant column would otherwise divide by zero
    rest = tuple((continuous_features(X, n_continuous) - mean) / std for X in other_X)
    return ((train_cont - mean) / std,) + rest


# UCI's own column order for the 10 continuous features -- used only for
# human-readable labels in CorrelationPruning.dropped, never to reorder data.
CONTINUOUS_FEATURE_NAMES = (
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
)


@dataclass
class CorrelationPruning:
    """A fixed, population-level decision about which continuous features
    are redundant -- computed ONCE from the full population and reused
    across every subsample/split seed. See module docstring: recomputing
    this per-subsample would let the pruned feature set itself vary by
    sample_seed, which breaks fi_drift's requirement that a generation-k and
    a generation-0 importance vector index the same columns, and would
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


def correlation_matrix(X, n_continuous=N_CONTINUOUS):
    """Pearson correlation matrix of the continuous features (see
    continuous_features()). Restricted to the continuous block for the same
    reason w2_feature is: correlating the one-hot Wilderness_Area/Soil_Type
    indicators would mostly just recover each block's built-in mutual
    exclusivity (columns of the same one-hot block are trivially
    anti-correlated by construction), not genuine feature redundancy.
    """
    return np.corrcoef(continuous_features(X, n_continuous), rowvar=False)


def highly_correlated_pairs(corr, threshold=0.9):
    """(i, j, r) triples of column indices with |corr[i, j]| >= threshold,
    i < j, most-correlated first. threshold=0.9 is the conventional
    multicollinearity cutoff; lower it to flag more aggressively. Returns
    raw indices, not names -- fit_correlation_pruning is what attaches
    CONTINUOUS_FEATURE_NAMES for a human-readable log."""
    p = corr.shape[0]
    pairs = [
        (i, j, float(corr[i, j]))
        for i in range(p) for j in range(i + 1, p)
        if abs(corr[i, j]) >= threshold
    ]
    pairs.sort(key=lambda t: -abs(t[2]))
    return pairs


def fit_correlation_pruning(threshold=0.9, cache_dir=DEFAULT_CACHE_DIR,
                             feature_names=CONTINUOUS_FEATURE_NAMES):
    """Decide which continuous features to drop as redundant, from the FULL
    population (fixed once -- see CorrelationPruning's docstring).

    Greedy findCorrelation-style heuristic: for each offending pair (|r| >=
    threshold, strongest first), drop whichever of the two features has the
    larger mean |correlation| against every OTHER continuous feature (not
    just its partner in this pair) -- a feature central to several redundant
    pairs is removed before one only weakly duplicated once. A feature
    already dropped by a stronger pair is left alone when a later, weaker
    pair involving it is considered.
    """
    X_full, _ = fetch_covertype_raw(cache_dir)
    corr = correlation_matrix(X_full)
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


def apply_pruning(X, pruning: CorrelationPruning, n_continuous=N_CONTINUOUS):
    """Drop the redundant continuous columns `pruning` identified from X,
    keeping the one-hot Wilderness_Area/Soil_Type block untouched and
    appended unchanged after the surviving continuous columns. Apply the
    SAME `pruning` object (fit once on the population) to every split --
    train, test, and any future tail-eval-equivalent -- never refit per
    split, for the reason CorrelationPruning's docstring explains.
    """
    cont = continuous_features(X, n_continuous)[:, pruning.keep_idx]
    onehot = X[:, n_continuous:]
    return np.hstack([cont, onehot])


if __name__ == "__main__":
    # python -m experiments.preprocessing.covertype -- relative imports
    # (.._common) need package context, same reason as every other module's
    # __main__ block in this project.
    from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

    geom = make_geometry()
    print(f"config_id: {geom.config_id()}  n_population={geom.n_population}")
    print("population class proportions:")
    for lbl, prop in zip(geom.class_labels, geom.class_props):
        flag = " <-- rare" if lbl in geom.rare_classes else ""
        print(f"  class {lbl}: {prop:.4f}{flag}")

    train, test = make_splits(geom, n_subsample=50_000, sample_seed=0)
    print(f"\ntrain: {train.summary()}")
    print(f"test:  {test.summary()}")

    clf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0,
    ).fit(train.X, train.y)
    pred = clf.predict(test.X)
    acc = float((pred == test.y).mean())
    rare_acc = float((pred[test.in_rare] == test.y[test.in_rare]).mean())
    print(f"\nbaseline RF: test acc={acc:.4f}  rare-class-proxy acc={rare_acc:.4f}")

    train_cont, test_cont = standardize_continuous(train.X, test.X)
    print(f"\nstandardized continuous features: train {train_cont.shape}, "
          f"test {test_cont.shape}, train mean~0: {train_cont.mean(axis=0).round(3)}")

    corr = correlation_matrix(train.X)
    print(f"\ncontinuous-feature correlation matrix ({len(CONTINUOUS_FEATURE_NAMES)}x"
          f"{len(CONTINUOUS_FEATURE_NAMES)}, from train):")
    for i, j, r in highly_correlated_pairs(corr, threshold=0.7):
        print(f"  {CONTINUOUS_FEATURE_NAMES[i]:35s} {CONTINUOUS_FEATURE_NAMES[j]:35s} r={r:+.3f}")

    pruning = fit_correlation_pruning(threshold=0.9)
    print(f"\nfit_correlation_pruning(threshold=0.9): config_id={pruning.config_id()}")
    if pruning.dropped:
        for name, partner, r in pruning.dropped:
            print(f"  dropped {name} (redundant with {partner}, r={r:+.3f})")
    else:
        print("  nothing dropped at this threshold")

    train_pruned = apply_pruning(train.X, pruning)
    test_pruned = apply_pruning(test.X, pruning)
    n_kept = len(pruning.keep_idx)
    clf_pruned = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0,
    ).fit(train_pruned, train.y)
    acc_pruned = float((clf_pruned.predict(test_pruned) == test.y).mean())
    print(f"\npruned: train {train_pruned.shape}, test {test_pruned.shape} "
          f"({n_kept}/{N_CONTINUOUS} continuous kept)")
    print(f"baseline RF on pruned features: test acc={acc_pruned:.4f} "
          f"(unpruned was {acc:.4f})")
