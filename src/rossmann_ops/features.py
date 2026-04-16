import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_features(
    df: pd.DataFrame,
    train_comp_median: float,
    expected_ohe_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Apply stateless feature transformations to a Rossmann DataFrame.

    Handles:
    - Date component extraction.
    - DayOfWeek standardization to 1-7 (Mon=1, Sun=7).
    - OHE encoding for StoreType, Assortment, StateHoliday.
    - Log-transformed CompetitionDistance (imputed with train-set median).
    - OHE column alignment: when expected_ohe_cols is provided, missing
      columns are filled with 0 and unexpected columns are dropped. This
      ensures inference payloads have the same feature space as training.

    NOTE: Does NOT compute Store_TargetMean. Call apply_target_encoding()
    separately after this function.

    Args:
        df: Input DataFrame (merged train or single-row inference payload).
        train_comp_median: CompetitionDistance median from the training set.
            Must be computed once from training data and reused at inference
            to prevent train/serve skew.
        expected_ohe_cols: Ordered list of OHE column names expected by the
            model. When provided, column alignment is enforced. Pass None
            during training (all categories are guaranteed present).

    Returns:
        Transformed DataFrame with engineered feature columns.
    """
    df = df.copy()

    # 1. Date Components
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
        # Standardize to 1=Monday, 7=Sunday (business convention)
        df["DayOfWeek"] = df["Date"].dt.dayofweek + 1

    # 2. Categoricals: OHE for StoreType, Assortment, StateHoliday
    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].astype(str)

    ohe_candidates = [
        c for c in ["StoreType", "Assortment", "StateHoliday"] if c in df.columns
    ]
    if ohe_candidates:
        df = pd.get_dummies(df, columns=ohe_candidates)

    # 3. Competition Distance: log-transform with train-median imputation
    if "CompetitionDistance" in df.columns:
        df["LogCompDist"] = np.log1p(
            df["CompetitionDistance"]
            .fillna(train_comp_median)
            .infer_objects(copy=False)
        )

    # 4. OHE Column Alignment (inference mode)
    # Ensures single-row payloads have the same feature space as training.
    if expected_ohe_cols is not None:
        for col in expected_ohe_cols:
            if col not in df.columns:
                df[col] = 0
        # Drop any unexpected OHE columns
        extra = [
            c
            for c in df.columns
            if any(
                c.startswith(p) for p in ("StoreType_", "Assortment_", "StateHoliday_")
            )
            and c not in expected_ohe_cols
        ]
        if extra:
            df = df.drop(columns=extra)

    logger.debug("build_features complete. Columns: %s", df.columns.tolist())
    return df


def apply_target_encoding(
    df: pd.DataFrame,
    store_means: dict[int, float],
    global_mean: float,
) -> pd.DataFrame:
    """
    Apply pre-computed Store-level target encoding to a DataFrame.

    Maps each Store ID to its historical mean sales. Falls back to the
    global training mean for unseen stores (new stores at inference time).

    Args:
        df: DataFrame containing a 'Store' column.
        store_means: Mapping of Store ID (int) to mean Sales (float).
            Computed from the training set and loaded from the JSON artifact
            at inference time.
        global_mean: Global mean Sales across all training stores.
            Used as fallback for stores not present in store_means.

    Returns:
        DataFrame with an added 'Store_TargetMean' column.
    """
    df = df.copy()
    df["Store_TargetMean"] = df["Store"].map(store_means).fillna(global_mean)
    logger.debug(
        "apply_target_encoding complete. Null count: %d",
        df["Store_TargetMean"].isna().sum(),
    )
    return df
