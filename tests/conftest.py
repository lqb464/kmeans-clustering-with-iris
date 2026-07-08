from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def raw_transactions():
    return pd.DataFrame(
        [
            {
                "Invoice": "1",
                "StockCode": "10001",
                "Description": "A",
                "Quantity": "2",
                "InvoiceDate": "2010-01-01",
                "Price": "10.0",
                "Customer ID": "1",
                "Country": "United Kingdom",
            },
            {
                "Invoice": "2",
                "StockCode": "10002",
                "Description": "B",
                "Quantity": 1,
                "InvoiceDate": "2010-01-10",
                "Price": 5.0,
                "Customer ID": 1,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "C3",
                "StockCode": "10001",
                "Description": "A",
                "Quantity": -1,
                "InvoiceDate": "2010-01-11",
                "Price": 10.0,
                "Customer ID": 1,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "4",
                "StockCode": "10003",
                "Description": "C",
                "Quantity": 1,
                "InvoiceDate": "bad",
                "Price": 8.0,
                "Customer ID": 2,
                "Country": "France",
            },
        ]
    )
