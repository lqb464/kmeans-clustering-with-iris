import numpy as np
import pandas as pd

from clustomer.models.artifact import load_artifact, save_artifact
from clustomer.models.trainers import train_kmeans


def sample_features(n=100):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Recency": np.r_[
                rng.integers(1, 20, n // 5),
                rng.integers(20, 80, n // 5),
                rng.integers(80, 300, 3 * n // 5),
            ],
            "Frequency": np.r_[
                rng.integers(10, 20, n // 5),
                rng.integers(3, 10, n // 5),
                rng.integers(1, 3, 3 * n // 5),
            ],
            "Monetary": np.r_[
                rng.uniform(3000, 8000, n // 5),
                rng.uniform(700, 3000, n // 5),
                rng.uniform(50, 700, 3 * n // 5),
            ],
        }
    )


def test_train_is_reproducible_and_review_rate_is_reasonable():
    features = sample_features()
    a1, s1 = train_kmeans(features, ("Recency", "Frequency", "Monetary"), 5, 42, 0.95)
    a2, s2 = train_kmeans(features, ("Recency", "Frequency", "Monetary"), 5, 42, 0.95)
    assert np.array_equal(s1.Cluster, s2.Cluster)
    assert 0.03 <= s1.ManualReview.mean() <= 0.07
    assert set(a1["segment_names"].values()) == {
        "Champions",
        "Loyal Growth",
        "New & Promising",
        "Needs Attention",
        "Dormant",
    }


def test_artifact_roundtrip(tmp_path):
    artifact, _ = train_kmeans(sample_features(), ("Recency", "Frequency", "Monetary"), 5, 42, 0.95)
    path = tmp_path / "model.joblib"
    checksum = save_artifact(artifact, path)
    loaded = load_artifact(path)
    assert len(checksum) == 64
    assert loaded["features"] == ["Recency", "Frequency", "Monetary"]


def test_aligned_training_preserves_locked_segment_names():
    features = sample_features()
    development, _ = train_kmeans(features, ("Recency", "Frequency", "Monetary"), 5, 42, 0.95)
    initial_centers_log = development["scaler"].inverse_transform(
        development["model"].cluster_centers_
    )
    locked_names = {cluster: name for cluster, name in development["segment_names"].items()}
    artifact, assignments = train_kmeans(
        features,
        ("Recency", "Frequency", "Monetary"),
        5,
        42,
        0.95,
        initial_centers_log=initial_centers_log,
        segment_names=locked_names,
    )
    assert artifact["segment_names"] == locked_names
    assert assignments["SegmentName"].notna().all()
