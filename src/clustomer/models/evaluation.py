from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score


def evaluate_clusters(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    shares = pd.Series(labels).value_counts(normalize=True)
    return {
        "silhouette": float(silhouette_score(matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
        "min_segment_share": float(shares.min()),
    }
