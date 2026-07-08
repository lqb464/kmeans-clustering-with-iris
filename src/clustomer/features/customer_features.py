from __future__ import annotations

import numpy as np
import pandas as pd


def build_customer_features(transactions: pd.DataFrame, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    data = transactions.loc[transactions["InvoiceDate"] <= cutoff].copy()
    if data.empty:
        raise ValueError("Không có giao dịch đủ điều kiện tại hoặc trước mốc chấm điểm")
    invoice = data.groupby(["Customer ID", "Invoice"], as_index=False).agg(
        InvoiceDate=("InvoiceDate", "min"),
        OrderValue=("LineRevenue", "sum"),
        BasketQuantity=("Quantity", "sum"),
    )
    features = invoice.groupby("Customer ID").agg(
        LastPurchase=("InvoiceDate", "max"),
        FirstPurchase=("InvoiceDate", "min"),
        Frequency=("Invoice", "nunique"),
        Monetary=("OrderValue", "sum"),
        AOV=("OrderValue", "mean"),
        AvgBasketSize=("BasketQuantity", "mean"),
    )
    features["Recency"] = (
        cutoff.normalize() + pd.offsets.Day(1) - features["LastPurchase"].dt.normalize()
    ).dt.days
    features["ActiveDays"] = (
        features["LastPurchase"].dt.normalize() - features["FirstPurchase"].dt.normalize()
    ).dt.days
    features["IsRepeat"] = features["Frequency"].gt(1).astype(int)
    features["ProductDiversity"] = data.groupby("Customer ID")["StockCode"].nunique()
    features["PurchaseCadence"] = np.where(
        features["IsRepeat"].eq(1),
        features["ActiveDays"] / (features["Frequency"] - 1),
        np.nan,
    )
    return features.drop(columns=["LastPurchase", "FirstPurchase"]).sort_index()
