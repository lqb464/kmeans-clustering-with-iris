from __future__ import annotations

import os
from datetime import datetime, timezone

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .evaluation import evaluate_clusters


def name_segments(features: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    profile = (
        features.assign(Cluster=labels)
        .groupby("Cluster")
        .agg(
            Recency=("Recency", "median"),
            Frequency=("Frequency", "median"),
            Monetary=("Monetary", "median"),
        )
    )
    value = (
        profile["Frequency"].rank(pct=True)
        + profile["Monetary"].rank(pct=True)
        - profile["Recency"].rank(pct=True)
    )
    remaining = set(profile.index)
    champions = int(value.idxmax())
    remaining.remove(champions)
    dormant = int(profile.loc[list(remaining), "Recency"].idxmax())
    remaining.remove(dormant)
    new = int(profile.loc[list(remaining), "Recency"].idxmin())
    remaining.remove(new)
    growth = int(profile.loc[list(remaining), "Frequency"].idxmax())
    remaining.remove(growth)
    attention = int(remaining.pop())
    return {
        champions: "Champions",
        growth: "Loyal Growth",
        new: "New & Promising",
        attention: "Needs Attention",
        dormant: "Dormant",
    }


def train_kmeans(
    features: pd.DataFrame,
    feature_names: tuple[str, ...],
    n_clusters: int,
    random_state: int,
    review_quantile: float,
    model_version: str = "1.0.0",
    initial_centers_log: np.ndarray | None = None,
    segment_names: dict[int, str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    selected = np.log1p(features.loc[:, feature_names])
    scaler = StandardScaler().fit(selected)
    matrix = scaler.transform(selected)
    if initial_centers_log is None:
        model = KMeans(n_clusters=n_clusters, n_init=30, random_state=random_state).fit(matrix)
    else:
        aligned_init = scaler.transform(pd.DataFrame(initial_centers_log, columns=feature_names))
        model = KMeans(
            n_clusters=n_clusters,
            init=aligned_init,
            n_init=1,
            random_state=random_state,
        ).fit(matrix)
    uncertainty = model.transform(matrix).min(axis=1)
    threshold = float(np.quantile(uncertainty, review_quantile))
    names = segment_names or name_segments(features, model.labels_)
    assignments = features.copy()
    assignments["Cluster"] = model.labels_
    assignments["SegmentName"] = assignments["Cluster"].map(names)
    assignments["AssignmentUncertainty"] = uncertainty
    assignments["ManualReview"] = uncertainty >= threshold
    artifact = {
        "model": model,
        "scaler": scaler,
        "features": list(feature_names),
        "segment_names": names,
        "uncertainty_threshold": threshold,
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": evaluate_clusters(matrix, model.labels_),
    }
    return artifact, assignments
