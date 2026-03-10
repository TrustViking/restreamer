from __future__ import annotations

import argparse


def build_cli_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Audit pipeline from Google Sheets to Google Docs and Telegram "
            "with shared preparation and branch execution."
        )
    )
    parser.add_argument(
        "--audit-mode",
        default="nomerge",
        metavar="{nomerge,merge,audit}",
        help="Run nomerge only, merge only, or audit (nomerge then merge).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send to Telegram and do not create Google Docs.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser
