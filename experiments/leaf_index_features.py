"""
Leaf-index recycling: the discrete counterpart to synthetic_features.py's
continuous OOB-prediction recycling.

experimental_design.md E3 ("the Ishwaran & Malley contrast") requires both
recycling schemes to run under an IDENTICAL generational loop:

    "They [Ishwaran & Malley] recycle continuous OOB predictions and find no
    performance loss; you recycle discrete leaf indices and find collapse.
    Run both under an identical loop. This directly tests your hypothesis
    that the discreteness/coarseness of the synthetic feature, not the mere
    fact of recycling, drives collapse."

This module is deliberately NOT structured like synthetic_features.py's
nodesize sweep -- Algorithm 1 sweeps D forests by construction, but "leaf
index" here means the per-tree terminal-node assignment of a single forest
(experimental_design.md 5's own example: `RandomForestClassifier.apply()`
plus `OneHotEncoder`). What both modules share, and what E3 actually needs,
is the recycling CONTRACT: `recycle(X, y, random_state, mode) -> X_next, y`,
so a loop driver can swap one module for the other without changing anything
else about the run.

Two encodings are implemented, per experimental_design.md 1.1's "strongest"
option -- report both and show the collapse pattern holds under both:

  - "ordinal": raw per-tree leaf-id integers, shape (n, ntree). This is the
    NOMINAL-AS-NUMERIC bug the design doc flags as the single biggest threat
    to validity (leaf 7 is not "between" leaf 6 and leaf 8). It is kept here
    on purpose, as a deliberate foil: if collapse appears under "ordinal" but
    not under "onehot", that is evidence of an encoding artifact, not model
    collapse, and needs to be reported as such rather than discovered after
    submission.
  - "onehot": per-tree leaf id one-hot encoded and concatenated across trees,
    via OneHotEncoder(sparse_output=True). Kept sparse throughout -- with
    ntree=500 and a few hundred leaves per tree this reaches ~10^5-10^6
    columns (experimental_design.md 5's own compute note), and densifying it
    is the single easiest way to exhaust memory on a laptop.

A fresh forest (and, for "onehot", a fresh encoder) is fit inside every call,
matching experimental_design.md 1's "fresh forest per generation is the
clean analog of 'train a new model on the previous model's output'" -- an
encoder fit on generation g's leaf ids is never reused at generation g+1.
"""

from dataclasses import dataclass, asdict
from typing import Union, Optional
import hashlib
import json

import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-untyped]

from ._common import child_seeds, resolve_mtry

VALID_ENCODINGS = ("ordinal", "onehot")


@dataclass
class LeafIndexConfig:
    """Fixed tuning choices for one leaf-index recycling step.

    ntree, mtry : same convention as SRFConfig (synthetic_features.py) and
        the paper's own "ntree=500, mtry=[p/3]" (experimental_design.md's
        insistence on an identical loop extends to using the same tuning
        rule across both recycling schemes, not just the same loop shape).
    nodesize    : single terminal-node size, mirroring the paper's plain
        "RF" baseline (nodesize=5) rather than Algorithm 1's sweep -- a
        single leaf partition is what "leaf-index recycling" means. Also the
        knob experimental_design.md E2 sweeps to control one-hot dimension
        count (`min_samples_leaf`).
    class_weight : passed straight to RandomForestClassifier. Needed for
        Covertype, whose two rare classes (~0.5% and ~1.6% of the
        population) are never predicted at all by an unweighted forest at
        this project's leaf granularity -- recall_rare sits at exactly 0.000
        from generation 0, which is a FLOOR, not a collapse measurement, and
        makes the rare-class metric uninformative for that dataset.
        "balanced" lifts it to ~0.88 and gives the metric headroom to
        degrade. Must be set on the RECYCLING forest and not only on the
        scoring forest: an unweighted partition drops the rare classes
        before the scoring forest ever sees the features.
    """
    ntree: int = 500
    mtry: Union[int, float, str] = "paper"
    nodesize: int = 5
    n_jobs: int = -1
    class_weight: Optional[str] = None

    def resolve_mtry(self, p):
        return resolve_mtry(self.mtry, p)

    def to_dict(self):
        return asdict(self)

    def config_id(self):
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:10]


def fit_leaf_forest(X, y, config=LeafIndexConfig(), random_state=0):
    """Fit one RF and return it with its per-sample, per-tree leaf ids.

    Returns
    -------
    clf      : fitted RandomForestClassifier
    leaf_ids : (n, ntree) int array, clf.apply(X) -- leaf_ids[i, t] is the
               terminal-node id sample i lands in under tree t.
    """
    if config.nodesize >= X.shape[0]:
        raise ValueError(
            f"nodesize={config.nodesize} >= n={X.shape[0]}: every tree would "
            f"be a single root leaf."
        )
    n_classes = len(np.unique(y))
    if n_classes < 2:
        raise ValueError(f"need >= 2 classes, got {n_classes}")

    mtry = config.resolve_mtry(X.shape[1])
    clf = RandomForestClassifier(
        n_estimators=config.ntree,
        min_samples_leaf=config.nodesize,
        max_features=mtry,
        bootstrap=True,
        class_weight=config.class_weight,
        n_jobs=config.n_jobs,
        random_state=random_state,
    ).fit(X, y)
    return clf, clf.apply(X)


def ordinal_leaf_features(leaf_ids):
    """Raw per-tree leaf ids as floats. Deliberately naive -- see module
    docstring's "ordinal" entry. Not one-hot: leaf id order carries no
    meaning, so a downstream tree's threshold split on this column is
    arbitrary."""
    return leaf_ids.astype(np.float64)


def onehot_leaf_features(leaf_ids, encoder=None):
    """One-hot per-tree leaf assignment, concatenated across trees, kept
    sparse (experimental_design.md 5: this can reach 10^5+ columns).

    Pass a previously-fit `encoder` to transform held-out data (e.g. the
    fixed real test set, or the tail-eval set) with the SAME categories the
    training forest actually produced; leave it None to fit a fresh one on
    training data.
    """
    if encoder is None:
        encoder = OneHotEncoder(categories="auto", handle_unknown="ignore",
                                 sparse_output=True)
        Z = encoder.fit_transform(leaf_ids)
    else:
        Z = encoder.transform(leaf_ids)
    return Z, encoder


def leaf_features(leaf_ids, encoding, encoder=None):
    """Dispatch to the requested encoding. `encoder` is ignored for
    "ordinal" and required (already-fit) when transforming held-out data
    under "onehot"."""
    if encoding not in VALID_ENCODINGS:
        raise ValueError(f"encoding must be one of {VALID_ENCODINGS}, got {encoding!r}")
    if encoding == "ordinal":
        return ordinal_leaf_features(leaf_ids), None
    return onehot_leaf_features(leaf_ids, encoder)


def transform_leaf_features(X_query, info, mode="replace"):
    """Score-time counterpart to recycle_leaf_indices, for held-out data
    (e.g. the fixed real test set) at the same generation. Same
    (X_query, info, mode) -> X_next_query contract as
    synthetic_features.transform_oob_predictions and
    random_partition_features.transform_random_partition, so a loop driver
    can score any of the three recycling schemes uniformly.

    Reuses info["leaf_forest"] (+ info["encoder"] for "onehot") from
    recycle_leaf_indices's return -- the training encoder's categories,
    rather than a freshly fit one, since a held-out set must be encoded
    consistently with what the generation's downstream classifier was
    actually trained on.
    """
    if mode not in ("replace", "accumulate"):
        raise ValueError(f"mode must be 'replace' or 'accumulate', got {mode!r}")
    if info["encoding"] == "onehot" and info["encoder"] is None:
        raise ValueError("info['encoder'] is required to transform held-out data under 'onehot'")

    leaf_ids = info["leaf_forest"].apply(X_query)
    Z, _ = leaf_features(leaf_ids, info["encoding"], info["encoder"])

    if mode == "replace":
        return Z
    if info["encoding"] == "ordinal":
        return np.hstack([X_query, Z])
    return sp.hstack([sp.csr_matrix(X_query), Z]).tocsr()


def recycle_leaf_indices(X, y, config=LeafIndexConfig(), random_state=0,
                          mode="replace", encoding="onehot"):
    """Produce the next generation's training features by recycling discrete
    leaf indices -- the E3 counterpart to recycle_oob_predictions in
    synthetic_features.py. Same (X, y, random_state, mode) -> (X_next, y)
    contract, so a loop driver can run both under identical conditions.

    mode="replace"    : X_next = Z only (primary condition).
    mode="accumulate" : X_next = [X, Z] (E4 comparison condition). Under
        "onehot" this returns a sparse matrix (X is densified into it);
        under "ordinal" it stays a small dense array.
    encoding="onehot" (default) or "ordinal" -- run BOTH across a study
        (experimental_design.md 1.1): if the collapse trajectory only
        appears under "ordinal", that is an encoding artifact, not evidence
        for the paper's claim.

    y is returned unchanged -- only features are regenerated each generation
    (experimental_design.md 1's scoping choice, shared with
    recycle_oob_predictions).

    Returns
    -------
    X_next, y, info : info holds `leaf_forest` and `encoder` (None for
        "ordinal") so the SAME generation's model can later be applied to
        held-out data via transform_leaf_features.
    """
    if mode not in ("replace", "accumulate"):
        raise ValueError(f"mode must be 'replace' or 'accumulate', got {mode!r}")

    leaf_forest, leaf_ids = fit_leaf_forest(X, y, config, random_state)
    Z, encoder = leaf_features(leaf_ids, encoding)

    if mode == "replace":
        X_next = Z
    elif encoding == "ordinal":
        X_next = np.hstack([X, Z])
    else:
        X_next = sp.hstack([sp.csr_matrix(X), Z]).tocsr()

    info = dict(leaf_forest=leaf_forest, encoder=encoder, encoding=encoding)
    return X_next, y, info


if __name__ == "__main__":
    # Relative imports (._common) require package context, so this demo must
    # be run as `python -m experiments.leaf_index_features` from the repo
    # root -- not `python experiments/leaf_index_features.py` directly. Same
    # reason datasets/ is imported as a package here rather than via a
    # sys.path hack: this is exactly the import a notebook at the repo root
    # would use.
    from datasets.generate_tier_a_dataset import make_geometry, make_splits

    geom = make_geometry()
    train, test, tail_eval = make_splits(geom)

    # Reduced ntree/larger nodesize for a fast demo only -- the real sweeps
    # use LeafIndexConfig()'s paper-matching defaults (ntree=500, nodesize=5),
    # which for one-hot easily reaches 10^5-10^6 columns per
    # experimental_design.md 5.
    demo_config = LeafIndexConfig(ntree=50, nodesize=50)
    print(f"config_id: {demo_config.config_id()}  n={len(train.y)}  "
          f"p={train.X.shape[1]}  classes={geom.n_classes}\n")

    for encoding in VALID_ENCODINGS:
        X_next, y_next, info = recycle_leaf_indices(
            train.X, train.y, demo_config, random_state=0,
            mode="replace", encoding=encoding,
        )
        shape = X_next.shape
        gen1 = RandomForestClassifier(
            n_estimators=demo_config.ntree,
            min_samples_leaf=demo_config.nodesize,
            max_features=demo_config.resolve_mtry(shape[1]),
            n_jobs=demo_config.n_jobs,
            random_state=0,
        ).fit(X_next, y_next)

        test_next = transform_leaf_features(test.X, info, mode="replace")
        print(f"{encoding:8s} X_next shape={shape}  "
              f"gen-1 train acc={gen1.score(X_next, y_next):.4f}  "
              f"gen-1 test acc={gen1.score(test_next, test.y):.4f}")
