from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from clustomer.config import load_settings
from clustomer.pipeline import SegmentPipeline

app = FastAPI(title="API phân khúc khách hàng Clustomer", version="1.0.0")


class PredictionRequest(BaseModel):
    cutoff: str = Field(description="Mốc chấm điểm theo định dạng ngày")
    transactions: list[dict] = Field(min_length=1, description="Danh sách các dòng giao dịch gốc")


@lru_cache(maxsize=1)
def resources():
    config_path = os.getenv("CLUSTOMER_CONFIG", "configs/config.yaml")
    settings = load_settings(config_path)
    return settings, SegmentPipeline(settings.model.artifact_path, settings.data.country)


@app.get("/health", summary="Kiểm tra trạng thái dịch vụ")
def health():
    try:
        settings, pipeline = resources()
        return {
            "status": "ok",
            "model_version": pipeline.artifact["model_version"],
            "project": settings.project["name"],
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/model", summary="Đọc metadata của mô hình")
def model_metadata():
    _, pipeline = resources()
    artifact = pipeline.artifact
    return {
        key: artifact.get(key)
        for key in ("model_version", "trained_at", "trained_through", "features", "metrics")
    }


@app.post("/predict", summary="Gán phân khúc cho giao dịch")
def predict(payload: PredictionRequest):
    settings, pipeline = resources()
    if len(payload.transactions) > settings.governance.max_payload_rows:
        raise HTTPException(status_code=413, detail="Payload vượt quá giới hạn số dòng đã cấu hình")
    try:
        result, warnings = pipeline.predict(pd.DataFrame(payload.transactions), payload.cutoff)
        result = result.where(pd.notna(result), None)
        return {
            "predictions": result.to_dict(orient="records"),
            "warnings": warnings,
            "model_version": pipeline.artifact["model_version"],
        }
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
