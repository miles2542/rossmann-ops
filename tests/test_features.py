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
    """DayOfWeek must be 1-7, not 0-6. 2026-04-08 is a Wednesday (3)."""
    df = build_features(_make_df(), train_comp_median=1270.0)
    assert df["DayOfWeek"].iloc[0] == 3


def test_log_comp_dist_present():
    """LogCompDist must be present and equal to log1p(CompetitionDistance)."""
    import numpy as np

    df = build_features(_make_df(CompetitionDistance=1000.0), train_comp_median=1000.0)
    assert "LogCompDist" in df.columns
    assert abs(df["LogCompDist"].iloc[0] - np.log1p(1000.0)) < 1e-6


def test_log_comp_dist_uses_train_median_for_nan():
    """NaN CompetitionDistance must be imputed with the provided train_comp_median."""
    import numpy as np

    median = 999.0
    df = build_features(_make_df(CompetitionDistance=None), train_comp_median=median)
    assert abs(df["LogCompDist"].iloc[0] - np.log1p(median)) < 1e-6


def test_ohe_columns_produced():
    """OHE should produce StoreType_a, Assortment_a, StateHoliday_0 columns."""
    df = build_features(_make_df(), train_comp_median=1270.0)
    assert "StoreType_a" in df.columns
    assert "Assortment_a" in df.columns
    assert "StateHoliday_0" in df.columns


def test_ohe_column_alignment_fills_missing():
    """When expected_ohe_cols provided, absent columns filled with 0."""
    expected = [
        "StoreType_a",
        "StoreType_b",
        "StoreType_c",
        "StoreType_d",
        "Assortment_a",
        "Assortment_b",
        "Assortment_c",
        "StateHoliday_0",
        "StateHoliday_a",
        "StateHoliday_b",
        "StateHoliday_c",
    ]
    df = build_features(
        _make_df(), train_comp_median=1270.0, expected_ohe_cols=expected
    )
    for col in expected:
        assert col in df.columns, f"Missing expected column: {col}"
    # StoreType_a should be 1, all others 0 (single store type a row)
    assert df["StoreType_a"].iloc[0] == 1
    assert df["StoreType_b"].iloc[0] == 0


def test_apply_target_encoding_known_store():
    """Known store ID must be mapped to its precomputed mean."""
    df = _make_df()
    store_means = {1: 6000.0, 2: 3000.0}
    result = apply_target_encoding(df, store_means, global_mean=4500.0)
    assert "Store_TargetMean" in result.columns
    assert result["Store_TargetMean"].iloc[0] == 6000.0


def test_apply_target_encoding_unknown_store_uses_global():
    """Unknown Store ID must fall back to global_mean."""
    df = _make_df(Store=[999])
    store_means = {1: 6000.0}
    result = apply_target_encoding(df, store_means, global_mean=4500.0)
    assert result["Store_TargetMean"].iloc[0] == 4500.0
