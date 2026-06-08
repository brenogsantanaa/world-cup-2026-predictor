"""Tests for chronological splitting (the leakage-safe train/test split)."""

import pandas as pd
import pytest

from sports_predictor.core.splitting import chronological_split, split_by_cutoff_date


def _frame():
    # Intentionally out of order so we can prove the split sorts first.
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-05", "2018-06-01", "2022-12-18", "2019-03-10", "2021-07-11"]
            ),
            "value": [50, 18, 22, 19, 21],
        }
    )


def test_split_puts_latest_rows_in_test():
    train, test = chronological_split(_frame(), test_fraction=0.4)

    assert len(train) == 3
    assert len(test) == 2
    # Every training date must be before every test date: no future leaks into train.
    assert train["date"].max() < test["date"].min()


def test_split_does_not_mutate_input_and_covers_all_rows():
    df = _frame()
    train, test = chronological_split(df, test_fraction=0.2)

    assert len(train) + len(test) == len(df)
    assert df.equals(_frame())  # input untouched


def test_invalid_fraction_raises():
    with pytest.raises(ValueError):
        chronological_split(_frame(), test_fraction=1.5)


def test_cutoff_split_is_strict():
    train, test = split_by_cutoff_date(_frame(), cutoff="2021-01-01")

    assert train["date"].max() < pd.Timestamp("2021-01-01")
    assert test["date"].min() >= pd.Timestamp("2021-01-01")
    assert len(train) == 3
    assert len(test) == 2
