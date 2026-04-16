import numpy as np
import pandas as pd

from rossmann_ops.features import apply_target_encoding, build_features


def _make_df(**overrides) -> pd.DataFrame:
    """Minimal valid DataFrame for feature engineering tests."""
    base = {
        "Date": ["2026-04-08"],
        "Store": [1],
        "Sales": [5000],
        "Promo": [1],
        "CompetitionDistance": [1270.0],
        "StoreType": ["a"],
        "Assortment": ["a"],
        "StateHoliday": ["0"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_day_of_week_is_one_indexed():
    """DayOfWeek must be 1-7 (Mon=1, Sun=7). 2026-04-08 is a Wednesday (3)."""
    df = build_features(_make_df(), train_comp_median=1270.0)
    assert df["DayOfWeek"].iloc[0] == 3


def test_log_comp_dist_math():
    """Assert LogCompDist = log1p(CompetitionDistance) with median imputation."""
    # Standard case
    df = build_features(_make_df(CompetitionDistance=1000.0), train_comp_median=500.0)
    assert abs(df["LogCompDist"].iloc[0] - np.log1p(1000.0)) < 1e-6

    # Imputation case: should use median=500 -> log1p(500)
    df_nan = build_features(_make_df(CompetitionDistance=None), train_comp_median=500.0)
    assert abs(df_nan["LogCompDist"].iloc[0] - np.log1p(500.0)) < 1e-6


def test_ohe_column_alignment_exact_schema():
    """
    Assert exact schema alignment:
    1. Missing categories in input are filled with 0.
    2. Unexpected categories in input are dropped.
    """
    expected = [
        "StoreType_a",
        "StoreType_b",
        "Assortment_a",
        "Assortment_b",
        "StateHoliday_0",
    ]
    # Input has StoreType 'c' which is NOT in expected
    df_input = _make_df(StoreType="c", Assortment="a", StateHoliday="0")

    df = build_features(df_input, train_comp_median=1270.0, expected_ohe_cols=expected)

    # 1. All expected columns must be present
    for col in expected:
        assert col in df.columns

    # 2. Unexpected StoreType_c must be dropped
    assert "StoreType_c" not in df.columns

    # 3. Values: StoreType_a/b should be 0 because input was 'c'
    assert df["StoreType_a"].iloc[0] == 0
    assert df["StoreType_b"].iloc[0] == 0
    assert df["Assortment_a"].iloc[0] == 1


def test_apply_target_encoding_fallback():
    """
    Assert known Store IDs get specific mean, and unknown Store ID (e.g. 99999)
    correctly falls back to global mean without crashing.
    """
    df = pd.DataFrame({"Store": [1, 99999]})
    store_means = {1: 6000.0, 2: 3000.0}
    global_val = 4500.0

    result = apply_target_encoding(df, store_means, global_mean=global_val)

    # Store 1 should be 6000
    assert result["Store_TargetMean"].iloc[0] == 6000.0
    # Store 99999 should be global_val
    assert result["Store_TargetMean"].iloc[1] == global_val
    # No NaNs allowed
    assert not result["Store_TargetMean"].isna().any()
