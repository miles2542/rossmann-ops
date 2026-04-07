from pydantic import BaseModel, Field
from datetime import date
from typing import Optional, List

class PredictRequest(BaseModel):
    """
    Schema for Rossmann Sales Prediction request.
    Transactional data provided by the user.
    """
    Store: int = Field(..., gt=0, description="The unique ID of the Rossmann store.")
    DayOfWeek: int = Field(..., ge=1, le=7, description="Day of the week (1=Monday, 7=Sunday).")
    Date: date = Field(..., description="The date for the forecast (YYYY-MM-DD).")
    Promo: int = Field(..., ge=0, le=1, description="Indicates whether a store is running a promo on that day.")
    StateHoliday: str = Field(..., description="Indicates a state holiday (a = public, b = Easter, c = Christmas, 0 = None).")
    # Optional override for competition if known, otherwise uses store dataset defaults
    CompetitionDistance: Optional[float] = Field(None, ge=0, description="Distance in meters to the nearest competitor store.")

    class Config:
        json_schema_extra = {
            "example": {
                "Store": 1,
                "DayOfWeek": 5,
                "Date": "2026-04-15",
                "Promo": 1,
                "StateHoliday": "0",
                "CompetitionDistance": 1270.0
            }
        }

class PredictResponse(BaseModel):
    """
    Schema for the prediction result.
    """
    Store: int
    Date: date
    PredictedSales: float
    ModelVersion: str

class DriftRequest(BaseModel):
    """
    Schema for drift trigger. List of recent sales observed.
    """
    sales_data: List[int]
