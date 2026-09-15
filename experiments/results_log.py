"""
Row-writer matching local-docs/results_log.csv's schema exactly, so
experiments.loop.run_seeds()'s output can be appended straight to the
tracker without hand-mapping column names or recomputing the
relative-to-gen0 ratios by hand.

Column order and names are copied VERBATIM from
local-docs/results_log.csv's header, so this module and that file cannot
silently drift apart:

    exp_id, config_id, dataset, feature_mode, d, rho, heavy_tail,
    min_samples_leaf, replace_mode, sample_schedule, synthetic_frac,
    structure_seed, sample_seed, generation, [the 12 metrics.csv columns],
    acc_rel_gen0, minrec_rel_gen0, notes

acc_rel_gen0/minrec_rel_gen0 are computed PER TRAJECTORY (per sample_seed):
each generation's acc_test/recall_minority divided by THAT SAME seed's own
generation-0 value -- not a value shared across seeds, because Xu et al.'s
point (experimental_design.md 3.3) is that typical and average trajectories
diverge, so even the baseline a trajectory is judged against should be its
own, not a pooled one that would understate how far a lucky/unlucky seed
actually moved.

d/rho/heavy_tail are Tier A's Geometry parameters and are left None (blank
on CSV write) for Tier B runs -- 0/0.0/False would silently assert false
things about real data ("rho=0" claims a real dataset has no induced
correlation parameter, which isn't a meaningful statement about it at all),
whereas blank correctly says "not applicable."

This module never writes to local-docs/results_log.csv on its own -- that
file is gitignored and hand-curated (its own row 2 is literally marked
"EXAMPLE ROW -- delete before your real run"), so silently appending to it
would risk corrupting notes someone is editing by hand. build_log_rows()
returns DataFrames; append_to_csv() is available for when a notebook
explicitly opts into writing one to a file.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd  # type: ignore[import-untyped]

from .metrics import GENERATION_METRIC_COLUMNS

RESULTS_LOG_COLUMNS = (
    ("exp_id", "config_id", "dataset", "feature_mode", "d", "rho", "heavy_tail",
     "min_samples_leaf", "replace_mode", "sample_schedule", "synthetic_frac",
     "structure_seed", "sample_seed", "generation")
    + GENERATION_METRIC_COLUMNS
    + ("acc_rel_gen0", "minrec_rel_gen0", "notes")
)


@dataclass
class RunConfig:
    """Run-level metadata broadcast onto every row produced by
    build_log_rows() for one run -- everything in results_log.csv's schema
    that is NOT a per-generation metric and not already on each row from
    experiments.loop.run_seeds() ("sample_seed", "generation").

    feature_mode should name the recycling scheme AND encoding together
    (e.g. "leaf_onehot", "leaf_ordinal", "oob", "random_onehot"), matching
    the RecyclingScheme.name values experiments.loop's *_scheme() factories
    already produce.
    """
    exp_id: str
    config_id: str
    dataset: str          # "tierA", "tierB_covertype", "tierB_income"
    feature_mode: str
    min_samples_leaf: int
    replace_mode: str      # "replace" or "accumulate"
    sample_schedule: str = "fixed"   # only "fixed" is in scope; the growing-n
                                     # sweep (former E5) was cut from the design
    synthetic_frac: float = 1.0       # E6's mixing fraction; 1.0 outside E6
    structure_seed: Optional[int] = None
    d: Optional[float] = None
    rho: Optional[float] = None
    heavy_tail: Optional[bool] = None
    notes: str = ""

    def to_dict(self):
        return asdict(self)


def _gen0_value(df: pd.DataFrame, column: str) -> pd.Series:
    """column's value at generation 0, looked up per sample_seed and
    broadcast back onto every row of that seed -- the per-trajectory
    baseline acc_rel_gen0/minrec_rel_gen0 divide by."""
    gen0 = df.loc[df["generation"] == 0].set_index("sample_seed")[column]
    return df["sample_seed"].map(gen0)


def build_log_rows(rows, run_config: RunConfig):
    """rows : experiments.loop.run_seeds()'s output (needs "seed" and
    "generation" keys -- run_trajectory() alone has no "seed" column, since
    a single trajectory doesn't need one; call this on run_seeds() output).

    Returns
    -------
    main  : DataFrame with EXACTLY RESULTS_LOG_COLUMNS, in that order --
        ready to concatenate with other runs' rows or append to
        local-docs/results_log.csv (see append_to_csv).
    extra : DataFrame with "sample_seed", "generation", and any columns in
        `rows` NOT part of RESULTS_LOG_COLUMNS (e.g. run_trajectory's
        extra_masks output -- "acc_race=Black" etc.), or None if there were
        none. results_log.csv's schema is fixed and has no room for
        per-group breakdowns, so these are kept and returned separately
        rather than silently dropped, or -- worse -- appended onto `main`
        and corrupting the file's column count against its existing header.
    """
    df = pd.DataFrame(rows)
    if "seed" not in df.columns or "generation" not in df.columns:
        raise ValueError(
            "rows must come from run_seeds() (needs 'seed' and 'generation' "
            "columns); run_trajectory() alone produces neither a 'seed' "
            "column nor rows comparable across seeds."
        )
    df = df.rename(columns={"seed": "sample_seed"})

    df["acc_rel_gen0"] = df["acc_test"] / _gen0_value(df, "acc_test")
    df["minrec_rel_gen0"] = df["recall_minority"] / _gen0_value(df, "recall_minority")

    for col, value in run_config.to_dict().items():
        df[col] = value

    for col in GENERATION_METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")  # not computed this run (e.g. no tail_eval on Tier B)

    extra_cols = [c for c in df.columns if c not in RESULTS_LOG_COLUMNS]
    extra = df[["sample_seed", "generation"] + extra_cols].copy() if extra_cols else None

    main = df[list(RESULTS_LOG_COLUMNS)]
    return main, extra


def append_to_csv(df: pd.DataFrame, path):
    """Append `df` (build_log_rows()'s `main` output) to an existing
    results_log.csv-shaped file at `path`, writing a header only if the
    file doesn't exist yet. Never called automatically by build_log_rows --
    see module docstring for why appending to the hand-curated tracker has
    to be something a notebook opts into explicitly."""
    path = Path(path)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


if __name__ == "__main__":
    # python -m experiments.results_log -- relative imports need package context.
    from datasets.generate_tier_a_dataset import make_geometry, make_splits
    from .loop import leaf_index_scheme, run_seeds
    from .leaf_index_features import LeafIndexConfig

    geom = make_geometry()
    train, test, tail_eval = make_splits(geom)

    scheme = leaf_index_scheme(LeafIndexConfig(ntree=50, nodesize=50), encoding="onehot")
    rows = run_seeds(
        train.X, train.y, test.X, test.y, scheme, n_generations=3, seeds=[0, 1],
        mode="replace", rf_kwargs=dict(n_estimators=50, min_samples_leaf=50, n_jobs=-1),
        in_rare=test.in_rare, in_tail=test.in_tail,
        tail_eval_X0=tail_eval.X, tail_eval_y=tail_eval.y,
    )

    run_config = RunConfig(
        exp_id="E1", config_id="E1-demo", dataset="tierA", feature_mode=scheme.name,
        min_samples_leaf=50, replace_mode="replace", structure_seed=0,
        d=geom.d, rho=geom.params["rho"], heavy_tail=geom.heavy_tail,
        notes="results_log.py demo run",
    )
    main, extra = build_log_rows(rows, run_config)

    print(f"main columns match RESULTS_LOG_COLUMNS: {tuple(main.columns) == RESULTS_LOG_COLUMNS}")
    print(f"main shape: {main.shape}\n")
    print(main[["sample_seed", "generation", "acc_test", "acc_rel_gen0",
                "recall_minority", "minrec_rel_gen0"]])

    gen0_ratio_ok = (main.loc[main["generation"] == 0, "acc_rel_gen0"] == 1.0).all()
    print(f"\nacc_rel_gen0 == 1.0 at every seed's generation 0: {gen0_ratio_ok}")
    print(f"extra (should be None -- no extra_masks used here): {extra}")
