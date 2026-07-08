from __future__ import annotations

import pandas as pd

from clustomer.cli import VietnameseArgumentParser, configure_utf8_stdout
from clustomer.monitoring.drift import population_stability_index


def main() -> int:
    configure_utf8_stdout()
    parser = VietnameseArgumentParser(
        description="So sánh phân phối đặc trưng khách hàng giữa dữ liệu tham chiếu và hiện tại"
    )
    parser.add_argument("reference", help="Tệp CSV tham chiếu")
    parser.add_argument("current", help="Tệp CSV hiện tại")
    args = parser.parse_args()
    reference, current = pd.read_csv(args.reference), pd.read_csv(args.current)
    common = [c for c in ("Recency", "Frequency", "Monetary") if c in reference and c in current]
    if not common:
        raise ValueError("Không tìm thấy cột đặc trưng cần giám sát")
    print(
        {
            column: population_stability_index(reference[column], current[column])
            for column in common
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
