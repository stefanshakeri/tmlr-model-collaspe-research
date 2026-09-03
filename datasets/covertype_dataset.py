"""
Covertype (UCI id=31): raw fetch + local cache only.

This module's one job is handing back the full 581,012-row population as
plain (X, y) numpy arrays, as cheaply as repeated calls allow. Subsampling,
train/test splitting, the rare-class proxy geometry, and continuous-feature
helpers for W2 all live in experiments/preprocessing/covertype.py instead --
this file doesn't know those exist. That split keeps "what raw data did we
get from UCI" (this file) separate from "what does an experiment need done
to it" (preprocessing), so a change to subsampling strategy or rare-class
threshold never touches the fetch/cache path and vice versa.
"""

from pathlib import Path

import numpy as np
from ucimlrepo import fetch_ucirepo  # type: ignore[import-untyped]

COVERTYPE_UCI_ID = 31
N_CONTINUOUS = 10  # Elevation .. Horizontal_Distance_To_Fire_Points; the
                    # remaining 44 columns are one-hot Wilderness_Area/Soil_Type.
                    # A property of the raw column layout UCI ships, not a
                    # preprocessing choice, so it's defined here.

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def fetch_covertype_raw(cache_dir=DEFAULT_CACHE_DIR):
    """The full 581,012-row Covertype population from UCI (id=31), fetched
    once via ucimlrepo and cached locally as .npz thereafter (a fresh fetch
    hits UCI's servers for a ~75MB payload every call, which is wasteful for
    something that never changes and blocks any offline run).

    Returns (X, y): X is (581012, 54) float64 (UCI's own encoding, unchanged
    -- 10 continuous columns first, then 4 Wilderness_Area + 40 Soil_Type
    one-hot indicators); y is (581012,) int64, 0-indexed (UCI's Cover_Type
    ships as 1-7).
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / "covertype_raw.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data["X"], data["y"]

    covertype = fetch_ucirepo(id=COVERTYPE_UCI_ID)
    X = covertype.data.features.to_numpy(dtype=np.float64)
    y = covertype.data.targets.to_numpy(dtype=np.int64).ravel() - 1

    if np.isnan(X).any():
        raise ValueError(
            "NaNs in Covertype features -- UCI's copy is expected to be "
            "complete; something upstream changed."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


if __name__ == "__main__":
    # python -m datasets.covertype_dataset
    X, y = fetch_covertype_raw()
    labels, counts = np.unique(y, return_counts=True)
    print(f"X {X.shape} {X.dtype}, y {y.shape} {y.dtype}")
    print("class counts:")
    for lbl, c in zip(labels, counts):
        print(f"  class {lbl}: {c} ({c / len(y):.4%})")
