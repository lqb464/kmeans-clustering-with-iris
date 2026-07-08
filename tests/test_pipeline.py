import pandas as pd

from clustomer.data.cleaner import clean_completed_purchases
from clustomer.data.schema import validate_and_coerce
from clustomer.features.customer_features import build_customer_features
from clustomer.models.artifact import save_artifact
from clustomer.models.trainers import train_kmeans
from clustomer.pipeline import SegmentPipeline


def test_raw_to_segment_pipeline_accepts_numeric_strings(tmp_path, raw_transactions):
    rows = []
    for customer in range(1, 31):
        for order in range(1, 1 + (customer % 6)):
            rows.append(
                {
                    "Invoice": f"{customer}-{order}",
                    "StockCode": f"{10000 + order}",
                    "Description": "item",
                    "Quantity": 1 + customer % 3,
                    "InvoiceDate": f"2010-01-{min(order + customer % 20, 28):02d}",
                    "Price": 2.0 + customer,
                    "Customer ID": customer,
                    "Country": "United Kingdom",
                }
            )
    training_raw, _ = validate_and_coerce(pd.DataFrame(rows))
    clean, _ = clean_completed_purchases(training_raw, "United Kingdom")
    features = build_customer_features(clean, "2010-01-31")
    artifact, _ = train_kmeans(features, ("Recency", "Frequency", "Monetary"), 5, 42, 0.95)
    path = tmp_path / "artifact.joblib"
    save_artifact(artifact, path)
    result, warnings = SegmentPipeline(path).predict(raw_transactions.iloc[:2], "2010-01-31")
    assert len(result) == 1
    assert result.loc[0, "SegmentName"] in artifact["segment_names"].values()
    assert result.loc[0, "ModelVersion"] == "1.0.0"
    assert warnings == []
