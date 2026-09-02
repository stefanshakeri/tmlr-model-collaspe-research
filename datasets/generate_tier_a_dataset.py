"""
Tier A synthetic dataset generator for the RF model-collapse experiments.

Design goal: every sample carries ground-truth metadata saying whether it is a
"tail" sample, under two independent definitions, so that tail loss across
generations is directly measurable rather than inferred.

Two tail definitions (deliberately redundant):
  1. Density-based  -- squared Mahalanobis distance from the class mean exceeds
     the chi-squared(d) quantile at `tail_quantile`. Exact, because the
     class-conditionals are Gaussian by construction.
  2. Cluster-based  -- membership in a small, deliberately placed rare
     subpopulation sitting well outside the class core.

If both definitions show the same collapse pattern, the finding is robust to
how "tail" was operationalised, which forecloses an obvious reviewer objection.

All sampling happens in whitened coordinates and is mapped through a Cholesky
factor at the end, so Euclidean distance in whitened space IS Mahalanobis
distance in feature space. That keeps the tail definition exact and makes
`separation` directly interpretable in standard-deviation units.

IMPORTANT -- structure and sampling seeds are deliberately separate:
`Geometry` (class means, covariance, rare-cluster centres) defines the
distribution and must be IDENTICAL across train / test / tail-eval and across
every generation of the recursive loop. Only `sample_seed` varies. Collapsing
these into one seed silently gives train and test unrelated class structure,
which still runs and still produces plausible-looking numbers while measuring
nothing at all.
"""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cholesky  # type: ignore[import-untyped]
from scipy.stats import chi2        # type: ignore[import-untyped]


@dataclass
class Geometry:
    """Fixed distributional structure. Build once, reuse everywhere."""
    d: int
    means: np.ndarray          # (n_classes, d) class means, whitened coords
    rare_centres: np.ndarray   # (n_classes, d) rare-cluster centres
    L: np.ndarray              # (d, d) Cholesky factor, whitened -> feature
    class_props: np.ndarray
    rare_prop: float
    rare_scale: float
    tail_quantile: float
    tail_threshold: float      # squared-Mahalanobis cutoff defining the tail
    heavy_tail: bool
    df: int
    params: dict = field(default_factory=dict)

    @property
    def n_classes(self):
        return len(self.class_props)

    def to_dict(self):
        """JSON-serialisable form. Save this alongside every results run --
        a config_id in the Results Log is worthless if the geometry it names
        cannot be reconstructed weeks later."""
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return out

    @classmethod
    def from_dict(cls, dct):
        dct = dict(dct)
        for k in ("means", "rare_centres", "L", "class_props"):
            dct[k] = np.asarray(dct[k])
        return cls(**dct)

    def config_id(self):
        """Stable short hash of the geometry. Use as `config_id` in the tracker."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:10]


@dataclass
class TierASplit:
    X: np.ndarray            # (n, d) features
    y: np.ndarray            # (n,) class labels
    in_tail: np.ndarray      # (n,) bool, density-based tail membership
    in_rare: np.ndarray      # (n,) bool, rare-subpopulation membership
    mahalanobis: np.ndarray  # (n,) distance from own-class mean

    def summary(self):
        return {
            "n": len(self.y),
            "d": self.X.shape[1],
            "class_balance": np.bincount(self.y).astype(float) / len(self.y),
            "tail_frac": float(self.in_tail.mean()),
            "rare_frac": float(self.in_rare.mean()),
        }


def make_geometry(
    d=10,
    class_props=(0.88, 0.12),
    rare_prop=0.015,
    rho=0.5,
    separation=3.0,
    rare_offset=3.2,
    rare_scale=0.30,
    tail_quantile=0.95,
    heavy_tail=False,
    df=3,
    structure_seed=0,
):
    """Define the distribution. Call ONCE per experimental configuration.

    Parameters
    ----------
    class_props : per-class proportions; imbalance is deliberate.
    rare_prop   : fraction of EACH class placed in a rare satellite cluster.
    rho         : AR(1) feature correlation. Non-zero on purpose -- axis-aligned
                  tree splits are genuinely lossy under correlation, which is
                  part of what the experiment probes. Sweep this.
    separation  : distance between class means in Mahalanobis units. Tune so
                  baseline accuracy AND minority recall both leave headroom to
                  degrade (see `calibrate` below).
    heavy_tail  : multivariate Student-t class conditionals (robustness check).
    """
    rng = np.random.default_rng(structure_seed)
    class_props = np.asarray(class_props, float)
    class_props = class_props / class_props.sum()
    n_classes = len(class_props)

    # Class means occupy frame columns 0..n_classes-2 and the rare-cluster offset
    # uses the last column. If they overlap, rare clusters land exactly along a
    # class-mean direction and stop being a separate tail structure -- which is
    # silent, not an error. Fail loudly instead.
    if n_classes > d - 1:
        raise ValueError(
            f"n_classes={n_classes} needs d >= {n_classes + 1} to keep class-mean "
            f"and rare-cluster directions distinct (got d={d})."
        )

    idx = np.arange(d)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L = cholesky(Sigma, lower=True)

    # Random orthonormal frame so no single raw feature trivially separates.
    frame = np.linalg.qr(rng.standard_normal((d, d)))[0]
    means = np.zeros((n_classes, d))
    for c in range(1, n_classes):
        means[c] = separation * frame[:, c - 1]

    # Rare clusters pushed out along an unused frame direction, so they land
    # deep in the density tail rather than between the classes.
    rare_centres = means + rare_offset * frame[:, -1]

    # Squared Mahalanobis distance is chi2(d)-distributed ONLY for Gaussian
    # conditionals. Under Student-t it is not, and chi2.ppf mislabels a large
    # fraction of the bulk as "tail" (empirically 28-46% instead of 5%).
    # So derive the cutoff empirically for the heavy-tailed case, once, here --
    # storing it on the Geometry keeps it identical across every split and
    # every generation of the recursive loop.
    if heavy_tail:
        ref_rng = np.random.default_rng(structure_seed + 999_983)
        ref = _noise(ref_rng, 200_000, d, True, df)
        tail_threshold = float(np.quantile((ref**2).sum(1), tail_quantile))
    else:
        tail_threshold = float(chi2.ppf(tail_quantile, d))

    return Geometry(
        d=d, means=means, rare_centres=rare_centres, L=L,
        class_props=class_props, rare_prop=rare_prop, rare_scale=rare_scale,
        tail_quantile=tail_quantile, tail_threshold=tail_threshold,
        heavy_tail=heavy_tail, df=df,
        params=dict(rho=rho, separation=separation, rare_offset=rare_offset,
                    structure_seed=structure_seed),
    )


def sample(geom: Geometry, n_samples, sample_seed):
    """Draw a split from a fixed Geometry. Only this seed should vary."""
    rng = np.random.default_rng(sample_seed)
    counts = rng.multinomial(n_samples, geom.class_props)

    Z_parts, y_parts, in_rare_parts = [], [], []
    for c, n_c in enumerate(counts):
        n_rare = int(round(n_c * geom.rare_prop))
        n_core = n_c - n_rare
        core = geom.means[c] + _noise(rng, n_core, geom.d, geom.heavy_tail, geom.df)
        rare = geom.rare_centres[c] + geom.rare_scale * rng.standard_normal((n_rare, geom.d))
        Z_parts.append(np.vstack([core, rare]))
        y_parts.append(np.full(n_c, c))
        in_rare_parts.append(np.r_[np.zeros(n_core, bool), np.ones(n_rare, bool)])

    Z = np.vstack(Z_parts)
    y = np.concatenate(y_parts)
    in_rare = np.concatenate(in_rare_parts)

    mahalanobis = np.linalg.norm(Z - geom.means[y], axis=1)
    in_tail = mahalanobis**2 > geom.tail_threshold

    perm = rng.permutation(len(y))
    return TierASplit(
        X=(Z @ geom.L.T)[perm], y=y[perm],
        in_tail=in_tail[perm], in_rare=in_rare[perm],
        mahalanobis=mahalanobis[perm],
    )


def _noise(rng, n, d, heavy_tail, df):
    g = rng.standard_normal((n, d))
    if not heavy_tail:
        return g
    return g / np.sqrt(rng.chisquare(df, size=(n, 1)) / df)  # Gaussian scale mixture


def make_splits(geom: Geometry, n_train=10_000, n_test=20_000,
                n_tail_eval=2_000, sample_seed=0):
    """Training set, fixed real test set, and a tail-enriched evaluation set.

    The tail-eval set exists because tail samples are only a few percent of the
    data: a plain 20k test set yields too few to estimate tail recall stably per
    generation. Build it once, report it separately, never fold it into the
    headline accuracy.

    The test sets are generated ONCE and must never be regenerated inside the
    recursive loop.
    """
    train = sample(geom, n_train, sample_seed)
    test = sample(geom, n_test, sample_seed + 10_000)

    pool = sample(geom, n_tail_eval * 40, sample_seed + 20_000)
    tail_idx = np.flatnonzero(pool.in_tail | pool.in_rare)
    take = np.random.default_rng(sample_seed).choice(
        tail_idx, size=min(n_tail_eval, len(tail_idx)), replace=False)
    tail_eval = TierASplit(
        X=pool.X[take], y=pool.y[take], in_tail=pool.in_tail[take],
        in_rare=pool.in_rare[take], mahalanobis=pool.mahalanobis[take],
    )
    return train, test, tail_eval


def calibrate(separations=(2.0, 3.0, 4.0, 5.0, 6.0), n_estimators=200, **geom_kw):
    """Baseline diagnostics at generation 0, before any recursive loop.

    What to look for: overall accuracy well above the majority-class rate, and
    minority recall / tail accuracy comfortably off both floor and ceiling.
    If minority recall starts near 0 the forest is just predicting the majority
    class and that metric can never show degradation; if everything starts near
    1.0 the task is too easy for collapse to be visible.
    """
    from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

    geom_kw.pop("separation", None)  # separation is the swept variable here
    rows = []
    for sep in separations:
        geom = make_geometry(separation=sep, **geom_kw)
        tr, te, tv = make_splits(geom)
        clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                     random_state=0).fit(tr.X, tr.y)
        pred = clf.predict(te.X)

        # Recall for EVERY class, and separately for the genuinely rarest one.
        # Hardcoding class 1 silently reports the middle class under a 3+ class
        # setup, which looks like a plausible number and is the wrong one.
        counts = np.bincount(te.y, minlength=geom.n_classes)
        per_class = {int(c): float((pred[te.y == c] == c).mean())
                     for c in range(geom.n_classes) if counts[c] > 0}
        rarest = int(np.argmin(np.where(counts > 0, counts, np.inf)))

        rows.append(dict(
            separation=sep,
            majority_rate=float(counts.max() / len(te.y)),
            accuracy=float((pred == te.y).mean()),
            rarest_class=rarest,
            rarest_class_recall=per_class[rarest],
            per_class_recall=per_class,
            tail_acc=float(clf.score(te.X[te.in_tail], te.y[te.in_tail])),
            rare_acc=float(clf.score(te.X[te.in_rare], te.y[te.in_rare])),
            tail_eval_acc=float(clf.score(tv.X, tv.y)),
        ))
    return rows


if __name__ == "__main__":
    print(f"config_id (default geometry): {make_geometry().config_id()}\n")
    print(f"{'sep':>5} {'major':>6} {'acc':>6} {'rareCls':>8} {'rareRec':>8} "
          f"{'tail':>6} {'rareSub':>8} {'tailEv':>7}")
    for r in calibrate():
        print(f"{r['separation']:5.1f} {r['majority_rate']:6.3f} {r['accuracy']:6.3f} "
              f"{r['rarest_class']:8d} {r['rarest_class_recall']:8.3f} "
              f"{r['tail_acc']:6.3f} {r['rare_acc']:8.3f} {r['tail_eval_acc']:7.3f}")