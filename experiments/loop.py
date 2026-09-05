"""
The recursive generational loop: fits a fresh forest each generation,
recycles its output into the next generation's training features via a
pluggable scheme, and scores the FIXED, never-regenerated real test set at
every generation -- the one piece every experiment in
local-docs/experimental_design.md (E1-E6) actually runs.

Design points, each answering a specific requirement from
experimental_design.md 1 ("Specify the loop formally before you run
anything"):

  - Fresh forest per generation: "the clean analog of 'train a new model on
    the previous model's output'." Every generation calls
    RandomForestClassifier(**rf_kwargs, random_state=...).fit(...) from
    scratch; nothing is carried over except the recycled features.
  - Only features are regenerated, never labels: run_trajectory reassigns
    `train_y` from each recycle call's return only for correctness against
    that contract (every recycle_* function returns y unchanged) -- it is
    never itself resampled.
  - replace vs accumulate is a run_trajectory PARAMETER (`mode`), not baked
    into a scheme, so E4 can compare both conditions under the identical
    scheme and seed.
  - Fixed n by default: nothing here grows n across generations (E5's
    schedule sweep needs that and is out of scope for this module).
  - The fixed test set, AND the tail-eval set (Tier A only), are pushed
    through the SAME per-generation transform chain as training data (via
    each scheme's `transform`), so at every generation g,
    compute_generation_metrics compares a classifier against evaluation
    data in EXACTLY the feature space that classifier was fit on. Missing
    this for tail_eval specifically is a real bug this module hit during
    its own first test run: clf.predict(tail_eval_X) failed at generation 1
    because tail_eval_X was still sitting in generation 0's feature space.
  - The ORIGINAL continuous features (`true_feature_train0`/`_test0`) are
    held fixed across every generation, entirely separate from whatever
    representation training is currently using -- mi_x_leaf and w2_feature
    ask how much information the CURRENT partition retains about the
    ORIGINAL data, not about this generation's recycled inputs.

RecyclingScheme is the adapter that makes E3's core comparison possible:
synthetic_features.recycle_oob_predictions, leaf_index_features.
recycle_leaf_indices, and random_partition_features.recycle_random_partition
are close to, but not exactly, the same call shape (only two of the three
take an `encoding`; only random-partition's transform needs fresh
randomness at score time). Forcing artificial uniformity onto those modules
would mean adding unused parameters to functions that don't need them; the
three *_scheme() factories below absorb the differences once, here, so
run_trajectory can call `scheme.recycle(X, y, seed, mode)` and
`scheme.transform(X_query, info, seed, mode)` identically regardless of
which of the three is active.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

from ._common import child_seeds
from .metrics import compute_generation_metrics, masked_accuracy
from .synthetic_features import SRFConfig, recycle_oob_predictions, transform_oob_predictions
from .leaf_index_features import LeafIndexConfig, recycle_leaf_indices, transform_leaf_features
from .random_partition_features import recycle_random_partition, transform_random_partition


@dataclass
class RecyclingScheme:
    """Adapter unifying one of the three recycle_*/transform_* module pairs
    into the single call shape run_trajectory needs:
        recycle(X, y, random_state, mode)          -> (X_next, y, info)
        transform(X_query, info, random_state, mode) -> X_next_query
    `random_state` is accepted (and ignored where irrelevant) by every
    scheme's transform so run_trajectory never needs to know which scheme is
    active to call it.
    """
    name: str
    recycle: Callable
    transform: Callable


def oob_scheme(config: SRFConfig = SRFConfig()) -> RecyclingScheme:
    """Ishwaran & Malley continuous OOB-prediction recycling (E3's
    published-result side)."""
    return RecyclingScheme(
        name="oob",
        recycle=lambda X, y, seed, mode: recycle_oob_predictions(X, y, config, seed, mode),
        transform=lambda Xq, info, seed, mode: transform_oob_predictions(Xq, info, mode),
    )


def leaf_index_scheme(config: LeafIndexConfig = LeafIndexConfig(),
                       encoding: str = "onehot") -> RecyclingScheme:
    """Discrete leaf-index recycling (E3's hypothesized-collapse side;
    E1's primary mechanism). encoding="ordinal" reproduces the
    experimental_design.md 1.1 encoding-artifact check."""
    return RecyclingScheme(
        name=f"leaf_index_{encoding}",
        recycle=lambda X, y, seed, mode: recycle_leaf_indices(X, y, config, seed, mode, encoding),
        transform=lambda Xq, info, seed, mode: transform_leaf_features(Xq, info, mode),
    )


def random_partition_scheme(config: LeafIndexConfig = LeafIndexConfig(),
                             encoding: str = "onehot") -> RecyclingScheme:
    """E2's control: a random partition of matched granularity, not learned
    from the data. Run under an identical loop to leaf_index_scheme with the
    same config/encoding to isolate generic information loss from anything
    RF-specific (experimental_design.md's E2)."""
    return RecyclingScheme(
        name=f"random_partition_{encoding}",
        recycle=lambda X, y, seed, mode: recycle_random_partition(X, y, config, seed, mode, encoding),
        transform=lambda Xq, info, seed, mode: transform_random_partition(Xq, info, seed, mode),
    )


def run_trajectory(
    train_X0, train_y, test_X0, test_y, scheme: RecyclingScheme,
    n_generations: int, random_state: int, mode: str = "replace",
    rf_kwargs: Optional[dict] = None,
    in_rare=None, in_tail=None,
    tail_eval_X0=None, tail_eval_y=None,
    true_feature_train0=None, true_feature_test0=None,
    track_fi_drift: bool = False,
    fi_method: str = "l1", mi_n_bins: int = 10, w2_max_samples: int = 1000,
    extra_masks: Optional[dict] = None,
):
    """Run ONE seed's recursive generational loop: fit a fresh forest, score
    it against the fixed test set (in this generation's own feature space),
    then recycle both train and test into the next generation's features.
    Produces n_generations + 1 rows (generations 0..n_generations
    inclusive).

    train_X0, train_y, test_X0, test_y : generation-0 data. test_X0/test_y
        must be the FIXED, never-regenerated real held-out set -- never draw
        a new one inside a loop.
    scheme       : one of oob_scheme()/leaf_index_scheme()/
                   random_partition_scheme() above.
    mode         : "replace" (primary condition) or "accumulate" (E4).
    rf_kwargs    : passed to RandomForestClassifier at every generation
                   (random_state is always overridden per-generation).
                   Defaults to this project's standard (ntree=500,
                   nodesize=5) if not given.
    in_rare, in_tail, tail_eval_X0/_y : forwarded to
        compute_generation_metrics every generation, unchanged -- these
        describe the FIXED test set (or, for tail_eval, a second fixed
        evaluation set) and never change across generations because y is
        never regenerated.
    true_feature_train0/_test0 : the ORIGINAL continuous features (Tier A's
        full X, or a Tier B continuous_features() slice), held fixed across
        every generation regardless of what train_X/test_X currently look
        like -- see module docstring.
    track_fi_drift : compute fi_drift against generation 0's
        feature_importances_. Only meaningful when the feature space is
        stable across generations (mode="accumulate", or a scheme with a
        fixed layout) -- see experiments.metrics.feature_importance_drift's
        docstring. Off by default because it is NOT meaningful under
        mode="replace" with leaf-index/OOB recycling, where the feature
        space changes identity every generation.
    extra_masks : optional {name: bool mask over test_y}, e.g. Income's
        IncomeSplit.group_masks (per-race, per-sex membership). Logs one
        extra column per mask, "acc_<name>", each generation's masked
        accuracy against the fixed test set in that generation's own
        feature space. This is what experimental_design.md's Income
        callout ("the minority-group representation metrics that connect
        directly to Wyllie et al.") actually needs -- compute_
        generation_metrics only threads through a single in_rare proxy,
        not a whole family of group breakdowns. Masks describe test-set ROW
        identity, which never changes across generations even though the
        FEATURE REPRESENTATION those rows are in does -- so the same masks
        apply unmodified at every generation.

    Returns
    -------
    list[dict] : one dict per generation, each metrics.csv column key ->
        value (whatever compute_generation_metrics could compute from the
        given inputs), each extra_masks key as "acc_<name>", plus a
        "generation" key. Feed to results_frame().
    """
    rf_kwargs = dict(rf_kwargs or {"n_estimators": 500, "min_samples_leaf": 5, "n_jobs": -1})

    # Two independent seed streams per generation (fit vs. recycle) rather
    # than reusing one seed for both -- consistent with every other module
    # in this project deriving independent children for independent random
    # draws, even though sklearn's and numpy's RNGs don't actually interact.
    seeds = child_seeds(random_state, 2 * (n_generations + 1))
    fit_seeds, recycle_seeds = seeds[0::2], seeds[1::2]

    rows = []
    X_train, X_test, tail_eval_X = train_X0, test_X0, tail_eval_X0
    baseline_importances = None

    for g in range(n_generations + 1):
        clf = RandomForestClassifier(random_state=fit_seeds[g], **rf_kwargs).fit(X_train, train_y)

        row = compute_generation_metrics(
            clf, X_train, X_test, test_y,
            tail_eval_X=tail_eval_X, tail_eval_y=tail_eval_y,
            in_rare=in_rare, in_tail=in_tail,
            baseline_importances=(baseline_importances if track_fi_drift else None),
            fi_method=fi_method,
            true_feature_train=true_feature_train0, true_feature_test=true_feature_test0,
            mi_n_bins=mi_n_bins, w2_max_samples=w2_max_samples, w2_random_state=fit_seeds[g],
        )
        row["generation"] = g

        if extra_masks:
            y_pred = clf.predict(X_test)
            for name, mask in extra_masks.items():
                row[f"acc_{name}"] = masked_accuracy(test_y, y_pred, mask)

        rows.append(row)

        if g == 0:
            baseline_importances = clf.feature_importances_

        if g < n_generations:
            X_train, train_y, info = scheme.recycle(X_train, train_y, recycle_seeds[g], mode)
            X_test = scheme.transform(X_test, info, recycle_seeds[g], mode)
            # tail_eval_X must ride the SAME per-generation transform chain
            # as X_test -- it's scored by this generation's clf too
            # (acc_tail_eval), so it has to live in the same feature space.
            if tail_eval_X is not None:
                tail_eval_X = scheme.transform(tail_eval_X, info, recycle_seeds[g], mode)

    return rows


def run_seeds(train_X0, train_y, test_X0, test_y, scheme: RecyclingScheme,
              n_generations: int, seeds, mode: str = "replace", **kwargs):
    """run_trajectory across many seeds, stacked into one flat row list --
    each row tagged with its "seed", ready for results_frame(). Running many
    independent seeds and reporting their distribution, not a mean curve, is
    the multi-seed requirement in experimental_design.md 3.3."""
    all_rows = []
    for seed in seeds:
        for row in run_trajectory(train_X0, train_y, test_X0, test_y, scheme,
                                   n_generations, seed, mode, **kwargs):
            row["seed"] = seed
            all_rows.append(row)
    return all_rows


def results_frame(rows) -> pd.DataFrame:
    """rows (from run_trajectory or run_seeds) as a DataFrame, one row per
    (seed, generation) -- input to trajectory_matrix, or the basis for a
    results_log.csv-schema row-writer built on top of this."""
    return pd.DataFrame(rows)


def trajectory_matrix(df: pd.DataFrame, metric: str) -> np.ndarray:
    """(n_seeds, n_generations+1) matrix of one metric's values, seed order
    ascending then generation order ascending -- the shape
    experiments.metrics.generations_to_threshold/
    probability_collapse_by_generation expect. `df` must have "seed" and
    "generation" columns (results_frame's output)."""
    return df.pivot(index="seed", columns="generation", values=metric).sort_index().to_numpy()


if __name__ == "__main__":
    # python -m experiments.loop -- relative imports need package context.
    from datasets.generate_tier_a_dataset import make_geometry, make_splits
    from .metrics import generations_to_threshold, probability_collapse_by_generation

    geom = make_geometry()
    train, test, tail_eval = make_splits(geom)

    # Small/fast config for a demo -- real runs use each scheme's own
    # paper-matching defaults.
    scheme = leaf_index_scheme(LeafIndexConfig(ntree=50, nodesize=50), encoding="onehot")
    n_generations, seeds = 5, [0, 1, 2]

    rows = run_seeds(
        train.X, train.y, test.X, test.y, scheme, n_generations, seeds,
        mode="replace", rf_kwargs=dict(n_estimators=50, min_samples_leaf=50, n_jobs=-1),
        in_rare=test.in_rare, in_tail=test.in_tail,
        tail_eval_X0=tail_eval.X, tail_eval_y=tail_eval.y,
    )
    df = results_frame(rows)
    print(f"{len(df)} rows = {len(seeds)} seeds x {n_generations + 1} generations\n")
    print(df.groupby("generation")[["acc_test", "recall_minority", "n_leaves_occupied"]].mean())

    recall_matrix = trajectory_matrix(df, "recall_minority")
    print(f"\nrecall_minority trajectory matrix shape: {recall_matrix.shape}")
    print(f"generations_to_threshold (frac=0.5) per seed: "
          f"{[generations_to_threshold(row) for row in recall_matrix]}")
    print(f"p_collapse_k (frac=0.5): {probability_collapse_by_generation(recall_matrix)}")
