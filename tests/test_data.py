import pandas as pd
import pytest
import pandera as pa
from src.data_validation import validate_data

def test_negative_sales_rejected():
    """Verify that negative sales values trigger a SchemaError."""
    df = pd.DataFrame({
        "Store": [1],
        "DayOfWeek": [1],
        "Sales": [-10],  # Invalid
        "Open": [1],
        "Promo": [1],
        "StateHoliday": ["0"],
        "SchoolHoliday": [0]
    })
    
    with pytest.raises(pa.errors.SchemaError):
        validate_data(df)

def test_missing_columns():
    """Verify that missing required columns trigger a SchemaError."""
    df = pd.DataFrame({
        "Store": [1],
        "Sales": [100]
        # Missing DayOfWeek, Open, etc.
    })
    
    with pytest.raises(pa.errors.SchemaError):
        validate_data(df)
