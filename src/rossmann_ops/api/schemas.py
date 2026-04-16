from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Schema for the Rossmann sales prediction request.
    Provides transactional context required to reconstruct the feature
    set used during training.
    """

    Store: int = Field(..., gt=0, description="Unique store ID.")
    DayOfWeek: int = Field(
        ..., ge=1, le=7, description="Day of week (1=Monday, 7=Sunday)."
    )
    Date: date = Field(..., description="Forecast date (YYYY-MM-DD).")
    Promo: int = Field(
        ..., ge=0, le=1, description="1 if a promo is active, 0 otherwise."
    )
    StateHoliday: str = Field(
        ...,
        description="State holiday type: 'a' (public), 'b' (Easter), 'c' (Christmas), '0' (none).",
    )
    StoreType: str = Field(
        ...,
        pattern="^[abcd]$",
        description="Store type category (a, b, c, or d).",
    )
    Assortment: str = Field(
        ...,
        pattern="^[abc]$",
        description="Assortment level (a=basic, b=extra, c=extended).",
    )
    CompetitionDistance: Optional[float] = Field(
        None,
        ge=0,
        description="Distance in meters to nearest competitor. Defaults to store dataset value.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Store": 1,
                "DayOfWeek": 5,
                "Date": "2026-04-15",
                "Promo": 1,
                "StateHoliday": "0",
                "StoreType": "a",
                "Assortment": "a",
                "CompetitionDistance": 1270.0,
            }
        }
    }


class PredictResponse(BaseModel):
    """Schema for the sales prediction response."""

    Store: int
    Date: date
    PredictedSales: float
    ModelVersion: str


class DriftRequest(BaseModel):
    """Schema for drift trigger. Accepts a list of recent observed sales values."""

    sales_data: List[int]
