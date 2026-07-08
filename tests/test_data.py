import pandas as pd
import pytest

from clustomer.data.cleaner import clean_completed_purchases
from clustomer.data.schema import SchemaError, validate_and_coerce


def test_schema_coerces_numeric_strings_and_reports_invalid_date(raw_transactions):
    data, warnings = validate_and_coerce(raw_transactions)
    assert pd.api.types.is_numeric_dtype(data["Quantity"])
    assert any("InvoiceDate" in warning for warning in warnings)


def test_schema_rejects_missing_column(raw_transactions):
    with pytest.raises(SchemaError):
        validate_and_coerce(raw_transactions.drop(columns="Price"))


def test_population_removes_return_and_non_uk(raw_transactions):
    data, _ = validate_and_coerce(raw_transactions)
    clean, audit = clean_completed_purchases(data, "United Kingdom")
    assert len(clean) == 2
    assert clean["Country"].eq("United Kingdom").all()
    assert (clean["Quantity"] > 0).all()
    assert audit.cancellation_rows == 1
