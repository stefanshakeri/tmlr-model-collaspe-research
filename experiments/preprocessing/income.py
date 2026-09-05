"""
Tier B preprocessing: Adult / Census Income, per
local-docs/experimental_design.md's Dataset selection > Tier B:

    "Adult / Census Income (~48k rows, UCI/OpenML). Class-imbalanced (~24%
    positive), contains genuinely heavy-tailed continuous features
    (capital-gain in particular), mixed categorical/continuous, and includes
    sensitive attributes -- which lets you report the minority-group
    representation metrics that connect directly to Wyllie et al. (2024).
    Small enough to iterate on a laptop."

That description is materially different from Covertype's, and this module
follows those differences rather than mechanically reusing Covertype's
shape:

  - No subsampling. Covertype is subsampled 581k -> ~50k because "there is
    no scientific reason to use all of it" (experimental_design.md 5); the
    doc gives no such instruction for Income (~48k, "small enough to iterate
    on a laptop" as-is), so make_splits here only cleans and splits the full
    population -- no n_subsample parameter.
  - in_rare is a RACE-rarity proxy, not a second target-class proxy. Income
    is binary (~24% positive), so a second "rare class" reading of in_rare
    would just duplicate recall_minority. What the doc actually calls out
    for Income specifically is sensitive attributes and minority-GROUP
    representation (Wyllie et al.) -- and race in this population has the
    same shape Covertype's classes do (population proportions: White 85.5%,
    Black 9.6%, Asian-Pac-Islander 3.1%, Amer-Indian-Eskimo 1.0%, Other
    0.8%), so in_rare = membership in a race category below `rare_threshold`
    is the direct Income analog of Covertype's rare-class proxy, applied to
    the attribute the doc actually asks for here.
  - group_masks additionally exposes EVERY race and sex category (not just
    the rare ones), because "the minority-group representation metrics" is
    plural and general in the doc's own phrasing -- Wyllie et al.'s point is
    tracking degradation across demographic groups broadly, not collapsing
    that down to a single scalar the way "how many classes forgotten" could
    for Covertype.
  - Missing values (workclass/occupation/native-country, ~2.5% of rows
    combined) are DROPPED, not imputed -- the doc specifies no imputation
    strategy, and inventing one would be an unstated modeling choice that
    could bias representation of exactly the sensitive-attribute groups this
    dataset exists to measure. Dropping is the standard, defensible choice
    for this dataset in the literature.
  - fnlwgt (the Census sampling weight) and the redundant string `education`
    column (education-num already encodes it numerically) are dropped before
    modeling -- standard Adult-dataset practice, not something the doc
    mandates but nothing in it argues against either.
  - capital-gain is left untouched (91.7% zero, heavy right tail, historical
    top-code at 99999) -- exactly the heavy-tailed feature the doc calls
    out, so standardize_continuous() below z-scores it for W2's Euclidean
    ground metric but never reshapes its distribution.
  - metrics.csv's Tier B column (acc_tail/acc_tail_eval "No", w2_feature
    "Limited" to the continuous block) applies here exactly as it does for
    Covertype -- see experiments/preprocessing/covertype.py's module
    docstring for the full rationale, not repeated here.

Structure/sample-seed split mirrors covertype.py: IncomeGeometry (population
class/race/sex proportions, the rare-race proxy set, and the FIXED one-hot
vocabulary for every categorical column) is computed ONCE from the full
cleaned population and reused across every split seed -- recomputing any of
it per split would let category vocabularies and "which race counts as
rare" drift with sampling noise, and would risk a split encountering a
category its one-hot columns don't have a slot for.

The raw fetch + local cache (fetch_income_raw) lives in
datasets/income_dataset.py; stratified sampling and correlation-based
pruning are shared with covertype.py via experiments/preprocessing/_shared.py.
See covertype.py's module docstring for why both splits are structured this
way.
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-untyped]

from .._common import child_seeds
from ._shared import (
    stratified_indices, CorrelationPruning, correlation_matrix as _correlation_matrix,
    highly_correlated_pairs, fit_correlation_pruning as _fit_correlation_pruning,
    apply_pruning as _apply_pruning,
)
from datasets.income_dataset import fetch_income_raw, DEFAULT_CACHE_DIR

# Fixed order: continuous columns first, then the one-hot categorical block.
# fnlwgt (sampling-weight artifact) and education (redundant with
# education-num) are deliberately excluded -- see module docstring.
CONTINUOUS_COLUMNS = (
    "age", "education-num", "capital-gain", "capital-loss", "hours-per-week",
)
CATEGORICAL_COLUMNS = (
    "workclass", "marital-status", "occupation", "relationship", "race",
    "sex", "native-country",
)
N_CONTINUOUS = len(CONTINUOUS_COLUMNS)


@dataclass
class IncomeSplit:
    """Real-data analog of TierBSplit (experiments/preprocessing/
    covertype.py) and Tier A's TierASplit. in_rare is the rare-race-group
    proxy (see module docstring); group_masks additionally exposes every
    race/sex category, not just the rare ones -- see module docstring for
    why Income's "minority-group representation" ask is broader than
    Covertype's "rare classes" one.
    """
    X: np.ndarray        # (n, N_CONTINUOUS + one-hot categorical columns)
    y: np.ndarray          # (n,) 0/1, 1 = income ">50K"
    in_rare: np.ndarray    # (n,) bool -- membership in an IncomeGeometry.rare_races category
    group_masks: dict      # {"race=Black": bool array, "sex=Female": bool array, ...}

    def summary(self):
        return {
            "n": len(self.y),
            "d": self.X.shape[1],
            "class_balance": np.bincount(self.y, minlength=2).astype(float) / len(self.y),
            "rare_frac": float(self.in_rare.mean()),
            "group_fracs": {k: float(v.mean()) for k, v in self.group_masks.items()},
        }


@dataclass
class IncomeGeometry:
    """Fixed structure: population class/race/sex proportions, the
    rare-race proxy set, and the one-hot vocabulary for every categorical
    column -- all computed once from the full cleaned population. Build
    once, reuse across every split seed -- see module docstring.
    """
    class_labels: np.ndarray    # [0, 1] -- 1 = income ">50K"
    class_props: np.ndarray
    race_labels: np.ndarray     # sorted race category strings
    race_props: np.ndarray
    rare_races: np.ndarray      # race labels with population proportion < rare_threshold
    rare_threshold: float
    sex_labels: np.ndarray
    sex_props: np.ndarray
    categorical_categories: dict  # {column: sorted category list} -- fixed one-hot vocabulary
    n_population: int             # after dropping rows with missing values
    n_dropped_missing: int

    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return out

    def config_id(self):
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:10]


def _clean_income_labels(y_raw):
    """UCI's Adult ships income labels as '<=50K'/'>50K' for the historical
    train partition and '<=50K.'/'>50K.' (trailing period) for the test
    partition -- an artifact of how the two source files were formatted,
    not a third and fourth class. Strip it, then map to 0/1."""
    return (y_raw.str.rstrip(".") == ">50K").to_numpy(dtype=np.int64)


def _load_clean(cache_dir):
    """Fetch + clean, shared by make_geometry and make_splits so the two can
    never disagree about which rows were dropped or how labels were
    normalized -- both must see the exact same cleaned population, row for
    row, or geom.categorical_categories could describe a different set of
    rows than what a split actually contains.

    Returns (df, y, n_dropped): df still has the 'race'/'sex' raw string
    columns (needed for group_masks) alongside CONTINUOUS_COLUMNS and
    CATEGORICAL_COLUMNS; y is 0/1; n_dropped counts rows removed for missing
    values.
    """
    X_df, y_raw = fetch_income_raw(cache_dir)
    df = X_df.copy()
    df["income"] = y_raw
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df)
    y = _clean_income_labels(df.pop("income"))
    return df, y, n_dropped


def make_geometry(rare_threshold=0.02, cache_dir=DEFAULT_CACHE_DIR):
    """Population class/race/sex proportions, the rare-race proxy set, and
    the fixed one-hot vocabulary (see module docstring). Default
    rare_threshold=0.02 isolates Amer-Indian-Eskimo (~1.0%) and Other
    (~0.8%) without also catching Asian-Pac-Islander (~3.1%) -- the same
    threshold Covertype uses, for the same reason: it is the value that
    isolates the population's genuinely small groups without also flagging
    its merely-smaller-than-majority ones.
    """
    df, y, n_dropped = _load_clean(cache_dir)

    labels, counts = np.unique(y, return_counts=True)

    race_labels, race_counts = np.unique(df["race"].to_numpy(), return_counts=True)
    race_props = race_counts / race_counts.sum()
    rare_races = race_labels[race_props < rare_threshold]

    sex_labels, sex_counts = np.unique(df["sex"].to_numpy(), return_counts=True)

    categorical_categories = {
        col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_COLUMNS
    }

    return IncomeGeometry(
        class_labels=labels, class_props=counts / counts.sum(),
        race_labels=race_labels, race_props=race_props,
        rare_races=rare_races, rare_threshold=rare_threshold,
        sex_labels=sex_labels, sex_props=sex_counts / sex_counts.sum(),
        categorical_categories=categorical_categories,
        n_population=len(df), n_dropped_missing=n_dropped,
    )


def _encode(df, geom):
    """Continuous columns first (raw scale, untouched), then the one-hot
    categorical block in geom's FIXED column order/vocabulary -- so every
    split produces an identical feature layout regardless of which
    categories happen to appear in that split. categories= (not "auto")
    is what makes this deterministic from geom alone rather than from
    whatever rows are passed in."""
    continuous = df[list(CONTINUOUS_COLUMNS)].to_numpy(dtype=np.float64)
    encoder = OneHotEncoder(
        categories=[geom.categorical_categories[c] for c in CATEGORICAL_COLUMNS],
        handle_unknown="ignore", sparse_output=False,
    )
    categorical = encoder.fit_transform(df[list(CATEGORICAL_COLUMNS)])
    return np.hstack([continuous, categorical])


def make_splits(geom, test_frac=0.2, sample_seed=0, cache_dir=DEFAULT_CACHE_DIR):
    """Stratified train/test split of the full cleaned population -- no
    subsampling (see module docstring). Stratified on the JOINT (income,
    race) key, not income alone: income-only stratification would preserve
    the ~24% positive rate but do nothing to protect the smallest race
    groups (Amer-Indian-Eskimo, Other -- a few hundred rows each) from
    landing unevenly between train and test by chance, which is exactly the
    representation this dataset is meant to let you measure.

    The test split is the fixed, never-regenerated real held-out set: draw
    it once per sample_seed and never call make_splits again for the same
    run (identical discipline to Tier A's and Covertype's make_splits).
    """
    df, y, _ = _load_clean(cache_dir)
    X = _encode(df, geom)

    strata = np.array([f"{yi}|{r}" for yi, r in zip(y, df["race"])])
    (split_seed,) = child_seeds(sample_seed, 1)
    n_test = int(round(len(y) * test_frac))
    test_idx = stratified_indices(strata, n_test, np.random.default_rng(split_seed))
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_idx] = True

    race_col = df["race"].to_numpy()
    sex_col = df["sex"].to_numpy()

    def _split(mask):
        groups = {f"race={r}": (race_col[mask] == r) for r in geom.race_labels}
        groups.update({f"sex={s}": (sex_col[mask] == s) for s in geom.sex_labels})
        return IncomeSplit(
            X=X[mask], y=y[mask],
            in_rare=np.isin(race_col[mask], geom.rare_races),
            group_masks=groups,
        )

    return _split(~test_mask), _split(test_mask)


def continuous_features(X, n_continuous=N_CONTINUOUS):
    """The first `n_continuous` columns -- the only ones
    experiments.metrics.wasserstein2_feature should be run on for Tier B.
    Defaults to all 5 raw continuous columns; pass len(pruning.keep_idx)
    after apply_pruning() has dropped some of them."""
    return X[:, :n_continuous]


def standardize_continuous(train_X, *other_X, n_continuous=N_CONTINUOUS):
    """Z-score the continuous columns using TRAIN statistics only, applied
    to train_X and every array in other_X (e.g. the fixed test split).
    Scales capital-gain/capital-loss for W2's Euclidean ground metric
    WITHOUT reshaping their distribution -- z-scoring is linear, so the
    heavy right tail the module docstring describes survives intact, only
    the units change. Never refit on anything but the fixed train split.
    """
    train_cont = continuous_features(train_X, n_continuous)
    mean = train_cont.mean(axis=0)
    std = train_cont.std(axis=0)
    std[std == 0] = 1.0  # a constant column would otherwise divide by zero
    rest = tuple((continuous_features(X, n_continuous) - mean) / std for X in other_X)
    return ((train_cont - mean) / std,) + rest


def correlation_matrix(X, n_continuous=N_CONTINUOUS):
    """Pearson correlation matrix of the continuous features (see
    continuous_features()). Restricted to the continuous block for the same
    reason w2_feature is: correlating the one-hot categorical indicators
    would mostly recover each block's own built-in mutual exclusivity, not
    genuine feature redundancy.
    """
    return _correlation_matrix(continuous_features(X, n_continuous))


def fit_correlation_pruning(threshold=0.9, cache_dir=DEFAULT_CACHE_DIR,
                             feature_names=CONTINUOUS_COLUMNS):
    """Decide which continuous features to drop as redundant, from the FULL
    cleaned population (fixed once -- CorrelationPruning's docstring in
    _shared.py explains why). Delegates the greedy findCorrelation-style
    heuristic to _shared.fit_correlation_pruning, shared with covertype.py;
    this wrapper's only job is fetching, cleaning, and slicing Income's own
    continuous block first.
    """
    df, _, _ = _load_clean(cache_dir)
    continuous = df[list(CONTINUOUS_COLUMNS)].to_numpy(dtype=np.float64)
    return _fit_correlation_pruning(continuous, feature_names, threshold)


def apply_pruning(X, pruning: CorrelationPruning, n_continuous=N_CONTINUOUS):
    """Drop the redundant continuous columns `pruning` identified from X,
    keeping the one-hot categorical block untouched and appended unchanged
    after the surviving continuous columns. Apply the SAME `pruning` object
    (fit once on the population) to every split -- never refit per split.
    """
    return _apply_pruning(X, pruning, n_continuous)


if __name__ == "__main__":
    # python -m experiments.preprocessing.income -- relative imports
    # (.._common, ._shared) need package context, same reason as every
    # other module's __main__ block in this project.
    from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

    geom = make_geometry()
    print(f"config_id: {geom.config_id()}  n_population={geom.n_population}  "
          f"n_dropped_missing={geom.n_dropped_missing}")
    print("population income balance:", dict(zip(geom.class_labels.tolist(), geom.class_props.round(4).tolist())))
    print("population race proportions:")
    for lbl, prop in zip(geom.race_labels, geom.race_props):
        flag = " <-- rare" if lbl in geom.rare_races else ""
        print(f"  {lbl:22s} {prop:.4f}{flag}")
    print("population sex proportions:",
          dict(zip(geom.sex_labels.tolist(), geom.sex_props.round(4).tolist())))

    train, test = make_splits(geom, sample_seed=0)
    print(f"\ntrain: {train.summary()}")
    print(f"test:  {test.summary()}")

    clf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0,
    ).fit(train.X, train.y)
    pred = clf.predict(test.X)
    acc = float((pred == test.y).mean())
    rare_acc = float((pred[test.in_rare] == test.y[test.in_rare]).mean())
    print(f"\nbaseline RF: test acc={acc:.4f}  rare-race-proxy acc={rare_acc:.4f}")
    print("per-group test accuracy:")
    for name, mask in test.group_masks.items():
        if mask.any():
            print(f"  {name:22s} n={int(mask.sum()):5d}  acc={float((pred[mask] == test.y[mask]).mean()):.4f}")

    train_cont, test_cont = standardize_continuous(train.X, test.X)
    print(f"\nstandardized continuous features: train {train_cont.shape}, "
          f"test {test_cont.shape}, train mean~0: {train_cont.mean(axis=0).round(3)}")

    corr = correlation_matrix(train.X)
    print(f"\ncontinuous-feature correlation matrix ({N_CONTINUOUS}x{N_CONTINUOUS}, from train):")
    pairs = highly_correlated_pairs(corr, threshold=0.3)
    if pairs:
        for i, j, r in pairs:
            print(f"  {CONTINUOUS_COLUMNS[i]:18s} {CONTINUOUS_COLUMNS[j]:18s} r={r:+.3f}")
    else:
        print("  no pair exceeds r=0.3")

    pruning = fit_correlation_pruning(threshold=0.9)
    print(f"\nfit_correlation_pruning(threshold=0.9): config_id={pruning.config_id()}")
    if pruning.dropped:
        for name, partner, r in pruning.dropped:
            print(f"  dropped {name} (redundant with {partner}, r={r:+.3f})")
    else:
        print("  nothing dropped at this threshold")
