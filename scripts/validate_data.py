from __future__ import annotations

from dataclasses import asdict

from clustomer.cli import VietnameseArgumentParser, configure_utf8_stdout
from clustomer.config import load_settings
from clustomer.data.cleaner import clean_completed_purchases
from clustomer.data.loader import load_transactions


def main() -> int:
    configure_utf8_stdout()
    parser = VietnameseArgumentParser(description="Kiểm tra dữ liệu giao dịch gốc của Clustomer")
    parser.add_argument("--config", default="configs/config.yaml", help="Đường dẫn cấu hình")
    args = parser.parse_args()
    settings = load_settings(args.config)
    data, warnings = load_transactions(settings.data.raw_path)
    clean, audit = clean_completed_purchases(data, settings.data.country)
    print({**asdict(audit), "customers": int(clean["Customer ID"].nunique()), "warnings": warnings})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
