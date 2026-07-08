from fastapi.testclient import TestClient

import backend.app as api


def test_predict_rejects_bad_schema(monkeypatch):
    class Settings:
        class governance:
            max_payload_rows = 10

    class Pipeline:
        artifact = {"model_version": "test"}

        def predict(self, frame, cutoff):
            raise ValueError("Thiếu các cột bắt buộc")

    monkeypatch.setattr(api, "resources", lambda: (Settings(), Pipeline()))
    response = TestClient(api.app).post(
        "/predict", json={"cutoff": "2010-01-01", "transactions": [{"bad": 1}]}
    )
    assert response.status_code == 422


def test_health_and_model_metadata(monkeypatch):
    class Settings:
        project = {"name": "Clustomer"}

    class Pipeline:
        artifact = {
            "model_version": "test",
            "trained_at": "now",
            "trained_through": "2010-10-31",
            "features": ["Recency", "Frequency", "Monetary"],
            "metrics": {"silhouette": 0.3},
        }

    monkeypatch.setattr(api, "resources", lambda: (Settings(), Pipeline()))
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/model").json()["features"] == ["Recency", "Frequency", "Monetary"]
