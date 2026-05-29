"""Out-of-time split + leakage tripwires.

The only honest test of a credit model: train on older origination vintages,
test on newer ones (never a random split — random splits leak the future into
the past via macro/vintage effects). A held-out **calibration** slice is carved
from the training vintages so probabilities are calibrated on data the models
never trained on (Part 3.2).

`assert_no_vintage_overlap` is the leakage tripwire enforced in tests + CI.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from creditforge.config import Config, load_config


@dataclass
class Split:
    train: pd.DataFrame
    calib: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> dict:
        def desc(d: pd.DataFrame) -> dict:
            return {
                "n": int(len(d)),
                "default_rate": float(d["default_12m"].mean()),
                "vintage_min": str(d["vintage"].min()),
                "vintage_max": str(d["vintage"].max()),
            }
        return {"train": desc(self.train), "calib": desc(self.calib),
                "test": desc(self.test)}


def _vintage_le(series: pd.Series, bound: str) -> pd.Series:
    return series <= bound        # 'YYYY-MM' strings sort chronologically

def _vintage_ge(series: pd.Series, bound: str) -> pd.Series:
    return series >= bound


def make_split(df: pd.DataFrame | None = None, cfg: Config | None = None) -> Split:
    cfg = cfg or load_config()
    if df is None:
        df = pd.read_parquet(cfg.path("gold") / "feature_matrix.parquet")

    oot = cfg.target.oot
    train_pool = df[_vintage_le(df["vintage"], oot.train_vintage_end)].copy()
    test = df[_vintage_ge(df["vintage"], oot.test_vintage_start)].copy()

    # Carve a calibration slice from the training pool (stratified by target),
    # seeded for reproducibility. Calibration must never touch test vintages.
    frac = float(oot.calib_fraction)
    calib = (train_pool.groupby("default_12m", group_keys=False)
             .sample(frac=frac, random_state=cfg.run.seed))
    train = train_pool.drop(index=calib.index)

    split = Split(train.reset_index(drop=True),
                  calib.reset_index(drop=True),
                  test.reset_index(drop=True))
    assert_no_vintage_overlap(split, cfg)
    return split


def assert_no_vintage_overlap(split: Split, cfg: Config | None = None) -> None:
    """Leakage tripwire: train/calib vintages must not reach into test vintages."""
    cfg = cfg or load_config()
    boundary = cfg.target.oot.test_vintage_start
    for name, frame in (("train", split.train), ("calib", split.calib)):
        if len(frame) and frame["vintage"].max() >= boundary:
            raise AssertionError(
                f"LEAKAGE: {name} contains vintage >= test boundary {boundary} "
                f"(max={frame['vintage'].max()})")
    if len(split.test) and split.test["vintage"].min() < boundary:
        raise AssertionError(
            f"LEAKAGE: test contains vintage < boundary {boundary} "
            f"(min={split.test['vintage'].min()})")


if __name__ == "__main__":
    import json
    s = make_split()
    print("[split]", json.dumps(s.summary(), indent=2))
