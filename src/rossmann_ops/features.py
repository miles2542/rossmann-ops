import logging

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, fill_value: float = None) -> pd.DataFrame:
    """
    Apply feature engineering to Rossmann dataset.
    - Extracts date components (Year, Month, Day, DayOfWeek, WeekOfYear).
    - Normalizes StateHoliday to string representation.
    - Fills CompetitionDistance using provided fill_value or calculated default.

    Args:
        df: Merged train and store DataFrame.
        fill_value: Static value to fill CompetitionDistance NaNs.
                   If None, calculates max * 2 from current data.

    Returns:
        DataFrame with engineered features.
    """
    logger.info("Building features...")
    df = df.copy()

    # 1. Date Extraction
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    # 2. StateHoliday handling
    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].astype(str)

    # 3. CompetitionDistance NaN handling
    if "CompetitionDistance" in df.columns:
        if fill_value is None:
            max_dist = df["CompetitionDistance"].max()
            fill_value = max_dist * 2 if not pd.isna(max_dist) else 100_000
            logger.info(f"Calculated dynamic competition fill: {fill_value}")

        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(fill_value)

    logger.info(
        "Feature engineering complete. Prepared Features: %s", df.columns.tolist()
    )
    return df
