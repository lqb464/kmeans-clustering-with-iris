"""Tiện ích chung cho giao diện dòng lệnh tiếng Việt."""

from __future__ import annotations

import argparse
import sys


class VietnameseArgumentParser(argparse.ArgumentParser):
    """ArgumentParser có tiêu đề và hướng dẫn trợ giúp bằng tiếng Việt."""

    def __init__(self, *args, **kwargs):
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self._positionals.title = "đối số vị trí"
        self._optionals.title = "tùy chọn"
        self.add_argument(
            "-h",
            "--help",
            action="help",
            default=argparse.SUPPRESS,
            help="Hiển thị hướng dẫn và thoát",
        )

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "cách dùng:", 1)

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "cách dùng:", 1)


def configure_utf8_stdout() -> None:
    """Bảo đảm terminal Windows có thể hiển thị tiếng Việt."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
