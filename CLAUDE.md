# Project Context for Claude Agent

## Session startup

At the start of each session, sync this file with the current code state before making other edits.

## What this project is

This is a production-grade daily content pipeline.
It ingests source links and metadata, groups content by date/time/language,
and publishes outputs to Google Docs and Telegram.

## Current architecture

Entry points: `promo.py`, `bot_main.py`
Main package: `app/`

| Package | Responsibility |
|---|---|
| `app/application` | Runtime orchestration |
| `app/bootstrap` | Startup checks, runtime setup, logging |
| `app/config` | Environment and YAML configuration loading |
| `app/core` | Shared models, constants, domain utilities, exceptions |
| `app/google` | Google Drive/Sheets/Docs clients |
| `app/ingest` | Source metadata extraction |
| `app/llm` | Merge pipeline, prompts, providers, quality normalization |
| `app/media` | Image helpers |
| `app/modes` | Single-item flow helpers |
| `app/net` | Shared HTTP client helpers |
| `app/observability` | Runtime analytics and summaries |
| `app/paths` | Name/path builders |
| `app/pipeline` | Batch and slot execution flow |
| `app/planning` | Planning, grouping, dedup |
| `app/publish` | Docs and Telegram publish layer |
| `app/resources` | Text and prompt resources |
| `app/telegram` | Telegram client |
| `app/telegram_bot` | Bot dispatcher and handlers |
| `app/tests` | Test suite |

## Operating modes

Runtime branch behavior is selected by CLI `--audit-mode`:
- `nomerge`
- `merge`
- `audit`

## LLM stack

- Provider abstraction in `app/llm/providers/`
- Structured merge schema ID: `merge_summary_v2`
- Multi-step flow: generate -> validate -> normalize -> fallback

## Naming status

All internal code uses neutral functional names.
No legacy project names remain in code symbols, strings, or identifiers.

## Priorities for next work sessions

3. Keep merge quality checks contract-focused and avoid semantic gate expansion.
4. Preserve output separation: publishable docs vs diagnostics in logs and JSON artifacts.
5. Add or adjust tests for behavior changes introduced by functional refactors.

## Developer context

- OS: Windows
- Virtual env: `.venv`
- Python style: explicit types, clear naming, single responsibility
