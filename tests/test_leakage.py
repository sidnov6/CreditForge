"""Leakage tripwires (Part 10.1 & 10.6).

The #1 way credit models lie is look-ahead. These tests assert the out-of-time
split never lets train/calib vintages reach into the test vintages.
"""
import pandas as pd
import pytest

from creditforge.config import load_config
from creditforge.pipeline.split import Split, assert_no_vintage_overlap


def _frame(vintages, n_each=20):
    rows = []
    for v in vintages:
        for i in range(n_each):
            rows.append({"vintage": v, "default_12m": i % 5 == 0})
    return pd.DataFrame(rows)


def test_clean_split_passes():
    cfg = load_config()
    boundary = cfg.target.oot.test_vintage_start  # e.g. "2020-01"
    split = Split(
        train=_frame(["2017-01", "2018-06"]),
        calib=_frame(["2019-01"]),
        test=_frame(["2020-03", "2021-01"]),
    )
    assert_no_vintage_overlap(split, cfg)  # must not raise


def test_train_reaching_into_test_is_caught():
    cfg = load_config()
    bad = Split(
        train=_frame(["2017-01", "2020-06"]),  # 2020-06 >= test boundary
        calib=_frame(["2019-01"]),
        test=_frame(["2020-03"]),
    )
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_vintage_overlap(bad, cfg)


def test_test_reaching_into_train_is_caught():
    cfg = load_config()
    bad = Split(
        train=_frame(["2017-01"]),
        calib=_frame(["2018-01"]),
        test=_frame(["2019-06", "2021-01"]),  # 2019-06 < boundary
    )
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_vintage_overlap(bad, cfg)
