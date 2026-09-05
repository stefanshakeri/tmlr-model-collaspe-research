"""
Adult / Census Income (UCI id=2): raw fetch + local cache only.

Same split as datasets/covertype_dataset.py: this module's one job is
handing back the ~48,842-row population as-is -- raw column names, raw
string categories, raw missing-value markers, raw duplicate-formatted income
labels -- with nothing cleaned, encoded, or dropped. All of that (missing
rows, the '<=50K.'/'>50K.' trailing-period quirk, one-hot encoding, the
sensitive-attribute group masks experimental_design.md calls for) is
experiments/preprocessing/income.py's job instead.
"""

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from ucimlrepo import fetch_ucirepo  # type: ignore[import-untyped]

ADULT_UCI_ID = 2

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def fetch_income_raw(cache_dir=DEFAULT_CACHE_DIR):
    """The full ~48,842-row Adult population from UCI (id=2), fetched once
    via ucimlrepo and cached locally thereafter. Cached as a pickle (not
    .npz like Covertype) because this data is genuinely mixed-type -- 6
    numeric columns, 8 string columns, and NaN missing markers -- and a
    DataFrame round-trips through pickle without needing pyarrow.

    Returns (X, y): X is a (48842, 14) DataFrame in UCI's own column order
    and dtypes, missing values as NaN (ucimlrepo already converts '?' on
    load); y is the (48842,) 'income' Series, UNNORMALIZED -- it still has
    four distinct raw values ('<=50K', '>50K', '<=50K.', '>50K.') because
    the original UCI train/test files used different trailing punctuation,
    not because there are four classes.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / "income_raw.pkl"
    if cache_path.exists():
        df = pd.read_pickle(cache_path)
        return df.drop(columns="income"), df["income"]

    adult = fetch_ucirepo(id=ADULT_UCI_ID)
    X = adult.data.features
    y = adult.data.targets.iloc[:, 0].rename("income")

    df = X.copy()
    df["income"] = y
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    return X, y


if __name__ == "__main__":
    # python -m datasets.income_dataset
    X, y = fetch_income_raw()
    print(f"X {X.shape}, columns: {list(X.columns)}")
    print(f"y {y.shape}, raw unique values: {sorted(y.unique())}")
    print(f"\nmissing values per column:\n{X.isna().sum()[X.isna().sum() > 0]}")
