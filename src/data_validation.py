import logging
from typing import Optional

import pandas as pd
import pandera.pandas as pa  # Updated to use pandas-specific classes as per warning
from pandera.typing import Series

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RossmannSchema(pa.DataFrameModel):
    """
    Pandera schema for Rossmann training data validation.
    Protects against data poisoning (e.g., negative sales) and schema drift.
    """

    Store: Series[int] = pa.Field(ge=1)
    DayOfWeek: Series[int] = pa.Field(ge=1, le=7)
    Sales: Series[int] = pa.Field(ge=0)
    Customers: Optional[Series[int]] = pa.Field(ge=0)
    Open: Series[int] = pa.Field(isin=[0, 1])
    Promo: Series[int] = pa.Field(isin=[0, 1])
    StateHoliday: Optional[Series[str]] = pa.Field(nullable=True)
    SchoolHoliday: Optional[Series[int]] = pa.Field(isin=[0, 1], nullable=True)

    class Config:
        """Schema configuration."""

        strict = False  # Allow extra columns (e.g., merged store data)
        coerce = True  # Automatically coerce types if possible


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate DataFrame against RossmannSchema.

    Args:
        df: Input DataFrame to validate.

    Returns:
        Validated DataFrame.

    Raises:
        pa.errors.SchemaError: If validation fails.
    """
    try:
        validated_df = RossmannSchema.validate(df)
        logger.info("Data validation successful. Row count: %d", len(validated_df))
        return validated_df
    except pa.errors.SchemaError as e:
        logger.error("Data validation failed: %s", str(error := e))
        raise error from None
