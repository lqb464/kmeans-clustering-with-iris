import numpy as np
import pandas as pd

from clustomer.data.cleaner import clean_completed_purchases
from clustomer.data.schema import validate_and_coerce
from clustomer.features.customer_features import build_customer_features


def test_customer_features_have_expected_rfm(raw_transactions):
    data, _ = validate_and_coerce(raw_transactions)
    clean, _ = clean_completed_purchases(data, "United Kingdom")
    features = build_customer_features(clean, "2010-01-31")
    assert features.loc[1, "Frequency"] == 2
    assert features.loc[1, "Monetary"] == 25
    assert features.loc[1, "Recency"] == 22
    assert np.isfinite(features[["Recency", "Frequency", "Monetary"]]).all().all()


def test_single_purchase_cadence_is_missing(raw_transactions):
    data, _ = validate_and_coerce(raw_transactions.iloc[[0]])
    clean, _ = clean_completed_purchases(data, "United Kingdom")
    features = build_customer_features(clean, "2010-01-31")
    assert np.isnan(features.iloc[0]["PurchaseCadence"])
    assert features.iloc[0]["IsRepeat"] == 0


def test_cutoff_prevents_future_leakage(raw_transactions):
    future = raw_transactions.iloc[[0]].copy()
    future["Invoice"] = "future"
    future["InvoiceDate"] = "2011-01-01"
    data, _ = validate_and_coerce(pd.concat([raw_transactions, future], ignore_index=True))
    clean, _ = clean_completed_purchases(data, "United Kingdom")
    features = build_customer_features(clean, "2010-01-31")
    assert features.loc[1, "Frequency"] == 2
    assert features.loc[1, "Monetary"] == 25
