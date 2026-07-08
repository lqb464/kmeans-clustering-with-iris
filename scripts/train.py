from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clustomer.cli import VietnameseArgumentParser, configure_utf8_stdout
from clustomer.config import load_settings
from clustomer.data.cleaner import clean_completed_purchases
from clustomer.data.loader import load_transactions
from clustomer.features.customer_features import build_customer_features
from clustomer.models.artifact import save_artifact
from clustomer.models.trainers import train_kmeans


def main() -> int:
    configure_utf8_stdout()
    parser = VietnameseArgumentParser(description="Huấn luyện thiết kế phân khúc Clustomer đã khóa")
    parser.add_argument("--config", default="configs/config.yaml", help="Đường dẫn cấu hình")
    args = parser.parse_args()
    settings = load_settings(args.config)
    raw, warnings = load_transactions(settings.data.raw_path)
    clean, _ = clean_completed_purchases(raw, settings.data.country)
    development_features = build_customer_features(
        clean, pd.Timestamp(settings.data.development_cutoff)
    )
    development_artifact, _ = train_kmeans(
        development_features,
        settings.model.features,
        settings.model.n_clusters,
        settings.model.random_state,
        settings.model.manual_review_quantile,
        settings.project["version"],
    )
    initial_centers_log = development_artifact["scaler"].inverse_transform(
        development_artifact["model"].cluster_centers_
    )
    segment_names = {
        int(cluster): name
        for cluster, name in json.loads(
            Path(settings.model.segment_names_path).read_text(encoding="utf-8")
        ).items()
    }
    features = build_customer_features(clean, pd.Timestamp(settings.data.validation_end))
    artifact, assignments = train_kmeans(
        features,
        settings.model.features,
        settings.model.n_clusters,
        settings.model.random_state,
        settings.model.manual_review_quantile,
        settings.project["version"],
        initial_centers_log=np.asarray(initial_centers_log),
        segment_names=segment_names,
    )
    artifact["trained_through"] = settings.data.validation_end
    artifact["population"] = (
        f"Khách hàng đã định danh tại {settings.data.country_display} "
        "với giao dịch mua hoàn tất và giá trị dương"
    )
    artifact["reference_statistics"] = {
        column: {
            "mean": float(features[column].mean()),
            "std": float(features[column].std()),
            "quantiles": [
                float(value) for value in features[column].quantile([0, 0.25, 0.5, 0.75, 1])
            ],
        }
        for column in settings.model.features
    }
    checksum = save_artifact(artifact, settings.model.artifact_path)
    assignments.to_csv("outputs/customer_segments.csv")
    print(
        {
            "customers": len(assignments),
            "sha256": checksum,
            "warnings": warnings,
            "metrics": artifact["metrics"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
