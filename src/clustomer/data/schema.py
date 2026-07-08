from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = (
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
)


class SchemaError(ValueError):
    """Lỗi khi dữ liệu giao dịch gốc vi phạm hợp đồng đầu vào."""


def validate_and_coerce(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise SchemaError(f"Thiếu các cột bắt buộc: {missing}")

    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    warnings: list[str] = []
    for column in ("Quantity", "Price", "Customer ID"):
        before = data[column].notna()
        data[column] = pd.to_numeric(data[column], errors="coerce")
        invalid = int((before & data[column].isna()).sum())
        if invalid:
            warnings.append(
                f"{column}: đã chuyển {invalid} giá trị không hợp lệ thành giá trị thiếu"
            )

    before_date = data["InvoiceDate"].notna()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    invalid_dates = int((before_date & data["InvoiceDate"].isna()).sum())
    if invalid_dates:
        warnings.append(
            f"InvoiceDate: đã chuyển {invalid_dates} giá trị không hợp lệ thành giá trị thiếu"
        )
    return data, warnings
