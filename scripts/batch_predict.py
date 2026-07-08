from __future__ import annotations

import pandas as pd

from clustomer.cli import VietnameseArgumentParser, configure_utf8_stdout
from clustomer.config import load_settings
from clustomer.pipeline import SegmentPipeline


def main() -> int:
    configure_utf8_stdout()
    parser = VietnameseArgumentParser(
        description="Gán phân khúc Clustomer cho tệp CSV giao dịch gốc"
    )
    parser.add_argument("input", help="Đường dẫn đến tệp CSV giao dịch gốc")
    parser.add_argument(
        "--output", default="outputs/batch_predictions.csv", help="Đường dẫn lưu kết quả"
    )
    parser.add_argument("--cutoff", required=True, help="Mốc chấm điểm theo định dạng ngày")
    parser.add_argument("--config", default="configs/config.yaml", help="Đường dẫn cấu hình")
    args = parser.parse_args()
    settings = load_settings(args.config)
    pipeline = SegmentPipeline(settings.model.artifact_path, settings.data.country)
    result, warnings = pipeline.predict(pd.read_csv(args.input), args.cutoff)
    result.to_csv(args.output, index=False)
    print({"customers": len(result), "output": args.output, "warnings": warnings})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
