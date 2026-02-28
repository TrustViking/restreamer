from __future__ import annotations

import argparse


def build_cli_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Batch pipeline from Google Sheets to Google Docs and Telegram "
            "with local thumbnail saving."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--merge",
        action="store_true",
        help="Run with LLM merge enabled for grouped descriptions.",
    )
    mode_group.add_argument(
        "--nomerge",
        action="store_true",
        help="Run without LLM merge; publish original titles/descriptions with numbering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send to Telegram and do not create Google Docs.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser
