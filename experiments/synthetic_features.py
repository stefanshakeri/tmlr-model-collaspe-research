"""
Synthetic feature generation, after Ishwaran & Malley (2014), "Synthetic
learning machines", Algorithm 1 (SRF).

Paper recap (Algorithm 1):
  1. Choose candidate nodesize values N = {n1, ..., nD}.
  2. Fit a RF with nodesize = nj for j = 1..D, same ntree/mtry throughout.
  3. The OOB predicted value of each RFj is its "synthetic feature".
  4. Fit a RF on [original features, synthetic features] -- the "synthetic RF".

Modifications made here, driven by local-docs/experimental_design.md:

  - Classification, not regression. Tier A (datasets/generate_tier_a_dataset.py)
    is a 2-4 class problem, so the synthetic feature per component forest is
    its OOB predicted class-probability vector with the last class dropped
    (paper Remark 1: J probabilities sum to 1, so J-1 encode them without
    redundancy).
  - OOB predictions only (paper Remark 2), enforced by raising if any row was
    never out-of-bag, rather than silently leaking in-sample fit.
  - ntree and mtry are fixed across the whole nodesize sweep AND reused,
    unchanged, for the final synthetic-RF fit in step 4 (paper Remark 3 and
    the algorithm's own "same ntree and mtry value as before"). mtry is
    computed from the ORIGINAL feature count and never recomputed against the
    augmented [X, Z] matrix -- recomputing it would silently violate that
    "same mtry" requirement.
  - Every random draw is derived from one `random_state` via
    `np.random.SeedSequence`, so a whole sweep (D component forests + the
    final fit) is reproducible from a single integer. This is required by
    experimental_design.md 3.3 (trajectory-level reproducibility across
    20-30+ seeds) and mirrors the structure/sample seed split already used in
    datasets/generate_tier_a_dataset.py.
  - `recycle_oob_predictions` exists on top of the paper's algorithm and is
    the actual point of this module for this project: experimental_design.md
    E3 pits Ishwaran & Malley's continuous OOB-prediction recycling (this
    module) against a discrete leaf-index recycling scheme (implemented
    elsewhere) run under an IDENTICAL generational loop. The paper only ever
    used the synthetic feature as a bolted-on addition to a single, static
    fit (step 4). Here it can instead become some or all of the NEXT
    generation's training features -- "replace" (the project's primary
    condition, matching Shumailov's full-resampling loop) or "accumulate"
    (the E4 comparison condition) -- while y is held fixed, since
    experimental_design.md 1 specifies only features are regenerated each
    generation, not labels.
"""

from dataclasses import dataclass, asdict
from typing import Union
import hashlib
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

from ._common import child_seeds, resolve_mtry


@dataclass
class SRFConfig:
    """Fixed tuning choices for one synthetic-forest sweep. Build once, reuse
    across every generation of a run so the sweep itself isn't a hidden
    source of nondeterminism.

    nodesize_grid : candidate terminal-node sizes N. Defaults to the paper's
        own choice (Results section, item 1): a handful of small values, a
        few intermediate, a few large, per Remark 4.
    ntree, mtry   : held fixed across the sweep and reused for the final
        synthetic-RF fit. mtry="paper" reproduces the paper's own
        mtry = [p/3] ("first integer greater than p/3"); pass an int, float,
        or sklearn's "sqrt"/"log2" to override.
    synthetic_nodesize : nodesize used for the final synthetic-RF fit
        (paper Results, item 1: fixed at 5 throughout their experiments).
    """
    nodesize_grid: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100)
    ntree: int = 500
    mtry: Union[int, float, str] = "paper"
    synthetic_nodesize: int = 5
    n_jobs: int = -1

    def resolve_mtry(self, p):
        """mtry as an int, given the original (pre-synthetic) feature count p."""
        return resolve_mtry(self.mtry, p)

    def to_dict(self):
        return asdict(self)

    def config_id(self):
        """Stable short hash, for tagging results the same way
        Geometry.config_id() tags a Tier A dataset configuration."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:10]


def _fit_oob_forest(X, y, nodesize, mtry, ntree, n_jobs, random_state):
    if nodesize >= X.shape[0]:
        raise ValueError(
            f"nodesize={nodesize} >= n={X.shape[0]}: every tree would be a "
            f"single root leaf. Shrink the nodesize grid or grow n."
        )
    clf = RandomForestClassifier(
        n_estimators=ntree,
        min_samples_leaf=nodesize,
        max_features=mtry,
        bootstrap=True,
        oob_score=True,
        n_jobs=n_jobs,
        random_state=random_state,
    ).fit(X, y)

    oob_proba = clf.oob_decision_function_
    if np.isnan(oob_proba).any():
        # A row is NaN in oob_decision_function_ only if it was never held out
        # of the bootstrap across all `ntree` trees -- vanishingly unlikely at
        # ntree=500 (paper Remark 3), but silently propagating NaNs into the
        # next generation's features would corrupt a collapse trajectory
        # without any visible symptom. Fail loudly instead.
        n_bad = int(np.isnan(oob_proba).any(axis=1).sum())
        raise RuntimeError(
            f"{n_bad} row(s) never out-of-bag at nodesize={nodesize} "
            f"(ntree={ntree}). Increase ntree."
        )
    return clf, oob_proba


def component_forests(X, y, config=SRFConfig(), random_state=0):
    """Algorithm 1, steps 1-3.

    Returns
    -------
    forests   : dict[nodesize -> fitted RandomForestClassifier]
    synthetic : dict[nodesize -> (n, J-1) OOB class-probability array],
                the paper's "synthetic feature" for that component forest.
    """
    n_classes = len(np.unique(y))
    if n_classes < 2:
        raise ValueError(f"need >= 2 classes, got {n_classes}")

    mtry = config.resolve_mtry(X.shape[1])
    seeds = child_seeds(random_state, len(config.nodesize_grid))

    forests, synthetic = {}, {}
    for nodesize, seed in zip(config.nodesize_grid, seeds):
        clf, oob_proba = _fit_oob_forest(
            X, y, nodesize, mtry, config.ntree, config.n_jobs, seed
        )
        forests[nodesize] = clf
        synthetic[nodesize] = oob_proba[:, :-1]  # drop last class (Remark 1)
    return forests, synthetic


def stack_synthetic_features(synthetic):
    """Concatenate the per-nodesize synthetic-feature blocks into one
    (n, D*(J-1)) matrix, in ascending-nodesize order so column layout is
    reproducible run to run."""
    order = sorted(synthetic)
    Z = np.hstack([synthetic[nodesize] for nodesize in order])
    return Z, order


def fit_synthetic_random_forest(X, y, config=SRFConfig(), random_state=0):
    """Algorithm 1 in full: steps 1-4. Recreates the paper's SRF exactly --
    the synthetic features augment, rather than replace, the original
    features, matching the paper's own definition of "synthetic RF"."""
    sweep_seed, final_seed = child_seeds(random_state, 2)
    forests, synthetic = component_forests(X, y, config, random_state=sweep_seed)
    Z, order = stack_synthetic_features(synthetic)
    X_augmented = np.hstack([X, Z])

    # Same mtry as the component sweep, computed from the ORIGINAL p -- not
    # recomputed against X_augmented's larger column count. See module
    # docstring: this is what "same mtry value as before" requires.
    mtry = config.resolve_mtry(X.shape[1])
    synthetic_rf = RandomForestClassifier(
        n_estimators=config.ntree,
        min_samples_leaf=config.synthetic_nodesize,
        max_features=mtry,
        bootstrap=True,
        oob_score=True,
        n_jobs=config.n_jobs,
        random_state=final_seed,
    ).fit(X_augmented, y)

    info = dict(component_forests=forests, synthetic_features=Z,
                nodesize_order=order, mtry=mtry)
    return synthetic_rf, info


def recycle_oob_predictions(X, y, config=SRFConfig(), random_state=0, mode="replace"):
    """Produce the next generation's training features by recycling
    continuous OOB predictions -- the E3 counterpart to a discrete
    leaf-index recycling scheme. Same
    (X, y, config, random_state, mode) -> (X_next, y, info) contract as
    leaf_index_features.recycle_leaf_indices and
    random_partition_features.recycle_random_partition, so a loop driver can
    swap between all three recycling schemes without changing anything else
    about the run.

    mode="replace"    : X_next = Z only (primary condition, full resampling;
                         experimental_design.md 1).
    mode="accumulate" : X_next = [X, Z] (E4 comparison condition).

    y is returned unchanged: only features are regenerated each generation
    (experimental_design.md 1's stated scoping choice).

    Returns
    -------
    X_next, y, info : info holds `component_forests` (the fitted RFj's --
        needed by transform_oob_predictions to score held-out data, e.g. the
        fixed real test set, at this same generation) and `nodesize_order`.
    """
    if mode not in ("replace", "accumulate"):
        raise ValueError(f"mode must be 'replace' or 'accumulate', got {mode!r}")

    forests, synthetic = component_forests(X, y, config, random_state)
    Z, order = stack_synthetic_features(synthetic)
    X_next = Z if mode == "replace" else np.hstack([X, Z])
    info = dict(component_forests=forests, nodesize_order=order)
    return X_next, y, info


def transform_oob_predictions(X_query, info, mode="replace"):
    """Score-time counterpart to recycle_oob_predictions, for held-out data
    (e.g. the fixed real test set) at the same generation. Reuses the SAME
    fitted component forests recorded in info["component_forests"] --
    scoring held-out rows uses ordinary predict_proba rather than an OOB
    prediction, since those rows were never part of any tree's bootstrap
    sample to begin with, so there is no leakage to guard against the way
    there is for training data (paper Remark 2).
    """
    if mode not in ("replace", "accumulate"):
        raise ValueError(f"mode must be 'replace' or 'accumulate', got {mode!r}")

    synthetic = {ns: clf.predict_proba(X_query)[:, :-1]
                 for ns, clf in info["component_forests"].items()}
    Z, _ = stack_synthetic_features(synthetic)
    return Z if mode == "replace" else np.hstack([X_query, Z])


if __name__ == "__main__":
    # Relative imports (._common) require package context, so this demo must
    # be run as `python -m experiments.synthetic_features` from the repo
    # root -- not `python experiments/synthetic_features.py` directly. Same
    # reason datasets/ is imported as a package here rather than via a
    # sys.path hack: this is exactly the import a notebook at the repo root
    # would use.
    from datasets.generate_tier_a_dataset import make_geometry, make_splits

    geom = make_geometry()
    train, test, tail_eval = make_splits(geom)
    config = SRFConfig()
    print(f"config_id: {config.config_id()}  n={len(train.y)}  "
          f"p={train.X.shape[1]}  classes={geom.n_classes}\n")

    # Plain RF at the paper's shared nodesize, as a baseline (paper's "RF").
    plain_rf = RandomForestClassifier(
        n_estimators=config.ntree, min_samples_leaf=config.synthetic_nodesize,
        max_features=config.resolve_mtry(train.X.shape[1]), n_jobs=config.n_jobs,
        random_state=0,
    ).fit(train.X, train.y)
    print(f"RF   test acc: {plain_rf.score(test.X, test.y):.4f}  "
          f"tail acc: {plain_rf.score(tail_eval.X, tail_eval.y):.4f}")

    # Full Algorithm 1 (paper's "SRF"): synthetic features augment the original.
    srf, srf_info = fit_synthetic_random_forest(train.X, train.y, config, random_state=0)
    test_augmented = transform_oob_predictions(test.X, srf_info, mode="accumulate")
    tail_augmented = transform_oob_predictions(tail_eval.X, srf_info, mode="accumulate")
    print(f"SRF  test acc: {srf.score(test_augmented, test.y):.4f}  "
          f"tail acc: {srf.score(tail_augmented, tail_eval.y):.4f}")

    # E3 hook: one recursive "replace" step -- next generation trains only on
    # recycled OOB predictions from generation 0, labels held fixed. The
    # fixed real test set is pushed through the SAME recycling step via
    # transform_oob_predictions, reusing generation 0's fitted component
    # forests -- exactly what a loop driver needs to score each generation.
    X_next, y_next, info = recycle_oob_predictions(
        train.X, train.y, config, random_state=0, mode="replace"
    )
    gen1_rf = RandomForestClassifier(
        n_estimators=config.ntree, min_samples_leaf=config.synthetic_nodesize,
        max_features=config.resolve_mtry(X_next.shape[1]), n_jobs=config.n_jobs,
        random_state=0,
    ).fit(X_next, y_next)
    test_next = transform_oob_predictions(test.X, info, mode="replace")
    print(f"\nrecycle_oob_predictions(mode='replace'): "
          f"X_next shape={X_next.shape}, nodesize_order={info['nodesize_order']}")
    print(f"gen-1 RF (trained on recycled OOB preds) train acc: "
          f"{gen1_rf.score(X_next, y_next):.4f}  test acc: "
          f"{gen1_rf.score(test_next, test.y):.4f}")
