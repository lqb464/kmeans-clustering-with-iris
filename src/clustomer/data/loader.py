from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import validate_and_coerce


def load_transactions(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    return validate_and_coerce(pd.read_csv(path))
