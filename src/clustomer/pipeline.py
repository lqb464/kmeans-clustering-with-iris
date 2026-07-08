from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data.cleaner import clean_completed_purchases
from .data.schema import validate_and_coerce
from .features.customer_features import build_customer_features
from .models.artifact import load_artifact


class SegmentPipeline:
    def __init__(self, artifact_path: str | Path, country: str = "United Kingdom"):
        self.artifact = load_artifact(artifact_path)
        self.country = country

    def predict(
        self, raw_transactions: pd.DataFrame, cutoff: str | pd.Timestamp
    ) -> tuple[pd.DataFrame, list[str]]:
        validated, warnings = validate_and_coerce(raw_transactions)
        clean, audit = clean_completed_purchases(validated, self.country)
        if clean.empty:
            raise ValueError("Không còn giao dịch nào sau khi kiểm tra quần thể")
        if audit.output_rows < audit.input_rows:
            warnings.append(f"Quy tắc quần thể giữ lại {audit.output_rows}/{audit.input_rows} dòng")
        features = build_customer_features(clean, cutoff)
        names = self.artifact["features"]
        matrix = self.artifact["scaler"].transform(np.log1p(features[names]))
        model = self.artifact["model"]
        labels = model.predict(matrix)
        if hasattr(model, "transform"):
            uncertainty = model.transform(matrix).min(axis=1)
        else:
            uncertainty = 1 - model.predict_proba(matrix).max(axis=1)
        result = features.loc[:, names].copy()
        result["Cluster"] = labels
        result["SegmentName"] = result["Cluster"].map(self.artifact["segment_names"])
        result["AssignmentUncertainty"] = uncertainty
        result["ManualReview"] = uncertainty >= self.artifact["uncertainty_threshold"]
        result["ModelVersion"] = self.artifact["model_version"]
        return result.reset_index(), warnings
