"""Label recycling: the condition that moves the loop from Alemohammad et
al.'s "synthetic loop with fixed real data" cell into the fully synthetic
cell.

Why this module exists
----------------------
Every other recycling scheme in this project (leaf_index_features,
synthetic_features, random_partition_features) regenerates FEATURES and
returns `y` unchanged, per experimental_design.md 1's scoping choice. That
choice has a consequence local-docs/results_to_paper.md 5.1 identifies as
the single biggest threat to the paper: ground-truth labels are re-supplied,
uncorrupted, at every generation, so real information never actually leaves
the loop. Degradation is bounded because an error at generation g cannot
become training signal at generation g+1 -- the label channel is closed.

experimental_design.md 1.2 names the corollary experiment, and it is this
one: open the label channel and see whether the collapse the feature-only
loop failed to produce appears. Two schemes, deliberately separating the two
channels rather than only running the combined condition, so the result can
attribute collapse to a channel rather than merely observe it:

  recycle_labels_only
      X stays the ORIGINAL features, unchanged, forever. Only y is
      regenerated, from the previous generation's forest. This is the clean
      isolation of the label channel and the closest analog in this project
      to Wyllie et al. (2024)'s MIDS setting, which recycles predicted
      labels rather than derived features.

  recycle_leaf_and_labels
      BOTH channels open: features become the leaf-index one-hot (exactly
      leaf_index_features' output) and labels become the previous
      generation's predictions. No component of the original data survives
      into generation g+1, which is what "fully synthetic loop" actually
      means in Alemohammad et al.'s taxonomy.

OOB predictions, not in-sample predictions
------------------------------------------
The recycled label is the out-of-bag predicted class, not `clf.predict(X)`.
This matters enough to be a correctness requirement rather than a
preference. A random forest's in-sample predictions on its own training
rows are near-perfect (each row was in ~63% of the bootstrap samples and the
trees memorise it), so an in-sample label loop would set y_next ~= y and
measure nothing -- it would report "no collapse" as an artifact of leakage.
The OOB prediction for row i uses only the trees that did NOT see row i,
which is the honest analog of "a model labelling data it has not been
trained on" and is the same discipline synthetic_features.py already
enforces for the OOB feature scheme (its Remark 2).

Rows that are never out-of-bag get no OOB vote. sklearn leaves those rows as
all-zero in `oob_decision_function_` (with a warning). Rather than raise --
which is what synthetic_features.py does, and which is right for a feature
matrix that would otherwise be silently wrong -- this module CARRIES THE
PREVIOUS GENERATION'S LABEL FORWARD for such rows. Dropping them would
shrink n across generations and confound the fixed-n design; raising would
make the module fail on exactly the small-ntree configurations the compute
budget forces. The count is returned in `info["n_no_oob"]` so a run can
report how often it happened, and with ntree >= 50 it is typically zero.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Union
import hashlib
import json
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

from ._common import resolve_mtry
from .leaf_index_features import LeafIndexConfig, leaf_features, transform_leaf_features


@dataclass
class LabelRecyclingConfig:
    """Tuning for the forest that GENERATES the next generation's labels.

    Deliberately mirrors LeafIndexConfig's fields and defaults so that the
    label-recycling condition and the leaf-index condition differ in what
    they recycle and nothing else -- the same requirement E3 imposes on the
    leaf-index/OOB comparison.
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


def oob_predicted_labels(X, y, config=LabelRecyclingConfig(), random_state=0):
    """The next generation's labels: each training row's out-of-bag predicted
    class under a forest fit on (X, y).

    Returns
    -------
    y_next   : (n,) int array of predicted classes, with rows that were never
               out-of-bag carrying their previous label forward (see module
               docstring).
    clf      : the fitted forest, so a caller can score held-out data with
               the same generation's model.
    n_no_oob : how many rows fell back to their previous label.
    """
    classes = np.unique(y)
    if len(classes) < 2:
        # The loop has already collapsed to a single surviving class. Fitting
        # is still possible but pointless, and oob_decision_function_ is
        # degenerate; return the constant labelling rather than pretend.
        return np.asarray(y).copy(), None, 0

    mtry = config.resolve_mtry(X.shape[1])
    with warnings.catch_warnings():
        # "Some inputs do not have OOB scores" is expected and handled below.
        warnings.simplefilter("ignore")
        clf = RandomForestClassifier(
            n_estimators=config.ntree,
            min_samples_leaf=config.nodesize,
            max_features=mtry,
            bootstrap=True,
            oob_score=True,
            class_weight=config.class_weight,
            n_jobs=config.n_jobs,
            random_state=random_state,
        ).fit(X, y)
        oob = clf.oob_decision_function_

    voted = np.nan_to_num(oob, nan=0.0).sum(axis=1) > 0
    y_next = np.asarray(y).copy()
    y_next[voted] = clf.classes_[np.nan_to_num(oob[voted], nan=0.0).argmax(axis=1)]
    return y_next, clf, int((~voted).sum())


def recycle_labels_only(X, y, config=LabelRecyclingConfig(), random_state=0,
                        mode="replace"):
    """Label channel ONLY: features pass through untouched, labels become the
    previous generation's OOB predictions.

    `mode` is accepted for RecyclingScheme call-shape compatibility and must
    be "replace". "accumulate" has no meaning here: there is no synthetic
    FEATURE block to append, and appending X to itself would just duplicate
    columns. Raising is better than silently ignoring the argument, since a
    caller passing mode="accumulate" has misunderstood the condition.

    Returns
    -------
    X, y_next, info : X is returned by identity -- generation g+1 trains on
        exactly the same feature matrix as generation 0, so any degradation
        is attributable to the labels alone.
    """
    if mode != "replace":
        raise ValueError(
            f"recycle_labels_only has no accumulate condition (see docstring); got mode={mode!r}"
        )
    y_next, clf, n_no_oob = oob_predicted_labels(X, y, config, random_state)
    info = dict(kind="labels_only", label_forest=clf, n_no_oob=n_no_oob)
    return X, y_next, info


def transform_labels_only(X_query, info, mode="replace"):
    """Score-time counterpart to recycle_labels_only: the identity.

    Held-out features are never re-encoded in this condition because
    training features are never re-encoded either. Note the test LABELS are
    untouched too -- run_trajectory never regenerates test_y, so every
    generation is still scored against the real held-out labels. That is the
    point: the training labels drift, and the fixed real test set measures
    how far.
    """
    if mode != "replace":
        raise ValueError(
            f"recycle_labels_only has no accumulate condition (see docstring); got mode={mode!r}"
        )
    return X_query


def recycle_leaf_and_labels(X, y, config=LeafIndexConfig(), random_state=0,
                            mode="replace", encoding="onehot",
                            label_config=None):
    """BOTH channels: leaf-index features AND OOB-predicted labels, so no
    component of the original data survives into the next generation. This is
    the fully-synthetic-loop cell of Alemohammad et al.'s taxonomy.

    One forest does both jobs, rather than fitting a separate forest per
    channel: the leaf partition and the OOB label vote come from the same
    fitted model, which is both cheaper and the more faithful analog of "the
    previous generation's model produced next generation's training set."
    `label_config` is therefore unused and accepted only so callers can pass
    the same keyword they would pass to the labels-only scheme.
    """
    if mode not in ("replace", "accumulate"):
        raise ValueError(f"mode must be 'replace' or 'accumulate', got {mode!r}")
    del label_config

    lab_cfg = LabelRecyclingConfig(ntree=config.ntree, mtry=config.mtry,
                                   nodesize=config.nodesize, n_jobs=config.n_jobs,
                                   class_weight=config.class_weight)
    y_next, clf, n_no_oob = oob_predicted_labels(X, y, lab_cfg, random_state)
    if clf is None:  # single surviving class -- keep the space fixed, stop re-encoding
        return X, y_next, dict(kind="leaf_and_labels", degenerate=True,
                               leaf_forest=None, encoder=None, encoding=encoding,
                               n_no_oob=n_no_oob)

    Z, encoder = leaf_features(clf.apply(X), encoding)
    if mode == "replace":
        X_next = Z
    elif encoding == "ordinal":
        X_next = np.hstack([X, Z])
    else:
        import scipy.sparse as sp
        X_next = sp.hstack([sp.csr_matrix(X), Z]).tocsr()

    info = dict(kind="leaf_and_labels", degenerate=False, leaf_forest=clf,
                encoder=encoder, encoding=encoding, n_no_oob=n_no_oob)
    return X_next, y_next, info


def transform_leaf_and_labels(X_query, info, mode="replace"):
    """Score-time counterpart to recycle_leaf_and_labels. Delegates to
    leaf_index_features.transform_leaf_features, which already reuses the
    training encoder's categories -- the same forest fitted in the recycle
    call is stored under info["leaf_forest"], so held-out rows are encoded by
    the generation's own model exactly as in the leaf-index condition.
    """
    if info.get("degenerate"):
        return X_query
    return transform_leaf_features(X_query, info, mode)
