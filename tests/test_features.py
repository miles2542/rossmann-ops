import pandas as pd

from rossmann_ops.features import build_features


def test_build_features_week_of_year():
    """Verify that build_features correctly extracts WeekOfYear from Date."""
    df = pd.DataFrame({
        "Date": ["2026-04-08"],
        "CompetitionDistance": [100]
    })

    df_out = build_features(df)

    assert "WeekOfYear" in df_out.columns
    # April 8, 2026 is Week 15
    assert df_out["WeekOfYear"].iloc[0] == 15

def test_build_features_fill_value():
    """Verify that passing a specific fill_value for CompetitionDistance is respected."""
    df = pd.DataFrame({
        "Date": ["2026-04-08"],
        "CompetitionDistance": [None]
    })

    # Custom fill value
    custom_fill = 99999.0
    df_out = build_features(df, fill_value=custom_fill)

    assert df_out["CompetitionDistance"].iloc[0] == custom_fill
