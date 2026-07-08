from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PopulationAudit:
    input_rows: int
    output_rows: int
    duplicate_rows: int
    missing_customer_rows: int
    non_country_rows: int
    cancellation_rows: int
    nonpositive_quantity_rows: int
    nonpositive_price_rows: int


def clean_completed_purchases(
    data: pd.DataFrame, country: str
) -> tuple[pd.DataFrame, PopulationAudit]:
    duplicate_rows = int(data.duplicated().sum())
    deduped = data.drop_duplicates().copy()
    missing_customer = deduped["Customer ID"].isna()
    non_country = ~deduped["Country"].eq(country)
    cancellation = deduped["Invoice"].astype(str).str.startswith("C")
    nonpositive_quantity = deduped["Quantity"].isna() | deduped["Quantity"].le(0)
    nonpositive_price = deduped["Price"].isna() | deduped["Price"].le(0)
    invalid_date = deduped["InvoiceDate"].isna()
    keep = ~(
        missing_customer
        | non_country
        | cancellation
        | nonpositive_quantity
        | nonpositive_price
        | invalid_date
    )
    clean = deduped.loc[keep].copy()
    clean["Customer ID"] = clean["Customer ID"].astype(int)
    clean["LineRevenue"] = clean["Quantity"] * clean["Price"]
    audit = PopulationAudit(
        input_rows=len(data),
        output_rows=len(clean),
        duplicate_rows=duplicate_rows,
        missing_customer_rows=int(missing_customer.sum()),
        non_country_rows=int(non_country.sum()),
        cancellation_rows=int(cancellation.sum()),
        nonpositive_quantity_rows=int(nonpositive_quantity.sum()),
        nonpositive_price_rows=int(nonpositive_price.sum()),
    )
    return clean.reset_index(drop=True), audit
