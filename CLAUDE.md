# restreamer — Project Context for Claude Agent

## What this project is

**restreamer** is a production-grade daily content pipeline.
It takes YouTube video links, extracts metadata, groups them by date/time/language,
optionally runs LLM-based merge summarization, and publishes results to Google Docs,
Google Sheets, and Telegram.

This is an **operational media production tool**, not an experiment or a script.
It runs daily. Reliability and clean output matter more than theoretical perfection.

---

## Current architecture

Entry point: `restreamer.py`
Main package: `app/`

| Package | Responsibility |
|---|---|
| `app/bootstrap` | Preflight, runtime context, startup health |
| `app/config` | `.env`-driven configuration loading |
| `app/ingest` | YouTube metadata extraction via `yt-dlp` |
| `app/planning` | Sheet loading, slot planning, merge rules, dedup |
| `app/pipeline` | Core orchestration and slot processing |
| `app/llm` | Merge service, quality gate, parser, polish, model/provider abstraction |
| `app/publish` | Google Docs writer, post-LLM sanitation |
| `app/telegram` | Renderer and batch sender |
| `app/observability` | Runtime analytics, summaries, health checks, usage tracking |
| `app/google` | Google Drive/Sheets API clients |
| `app/paths` | Name and path builders |
| `tests/` | Test suite |

**Key dependencies:** `yt-dlp`, `openai`, `google-api-python-client`, `openpyxl`, `pillow`, `python-dotenv`

---

## Operating modes

| Mode | Behavior |
|---|---|
| `nomerge` | Publish each video as-is, no LLM summarization |
| `merge` | LLM combines multiple sources into one editorial summary |
| `audit` | Run both branches for comparison |

Input comes from **Google Sheets** (links + schedule).
Output goes to **Google Docs** (human-readable) + **Telegram** (channel posts).

---

## Language handling

Videos are tagged as: `uk` (Ukrainian), `en` (English), `ru` (Russian), `other`.
Merge pipeline is language-aware. Outputs are structured per-language per-slot.

---

## Coding standards

- **Explicit type annotations** on all variables and parameters
- **Meaningful English names** — no single-letter abbreviations
- **Single responsibility per method**
- Dataclasses for domain models
- Explicit logging throughout
- Environment-driven config via `.env`

---

## LLM stack

- Provider abstraction in `app/llm/provider_factory.py`
- Current models: `gpt-4o` family (configured via `.env`)
- Local fallback option: Ollama (`gemma3:12b` primary, `gemma3:4b` fallback)
- Multi-step pipeline: generate → validate → retry → fallback

---

## Current state and known issues

### What works well
- Full batch pipeline from Sheets input to Docs + Telegram output
- Audit mode (nomerge vs merge comparison)
- Slot/language grouping
- Observability: run summaries, merge metrics, debug JSON artifacts
- Thumbnail/preview generation

### Active problem: LLM quality gate is over-engineered

The merge quality validator (`app/llm/merge_quality.py`) currently rejects
semantically valid model outputs based on subjective criteria:
`too_few_expanded_bullets`, `insufficient_expanded_body`, `overly_generic_body`,
`weak_source_coverage`, etc.

**This causes ~60–80% of merge candidates to fall back**, even when the model
produced usable editorial content.

The validator has grown beyond its intended role — it now acts as a second
editor trying to outjudge the model, rather than a technical contract checker.

### What the quality gate SHOULD do (technical contract only)
- Response parsed successfully ✓
- Required fields present and non-empty ✓
- Output format is publication-ready ✓

### What it should NOT do
- Evaluate semantic depth, topic spread, bullet count, body density
- Inject diagnostic annotations into the output document

### Debug output belongs in logs/JSON, not in the document
The output Google Doc currently shows: `Partial fallback`, `REJECTED TITLE`,
`MODEL OUTPUTS REJECTED BY VALIDATION`, etc.
**These must be removed from the doc entirely** — they belong in `debug/*.json` only.

---

## Priorities for next work sessions

1. **Strip semantic checks from the merge quality gate** — keep only structural/format validation
2. **Remove all diagnostic annotations from Google Docs output** — clean doc = clean result
3. **Simplify merge retry logic** — reduce retry profiles, remove targeted semantic retries
4. **Hard separation: doc = publishable result / JSON = full forensic data**
5. No new heuristics or validation rules until the above is done

---

## What NOT to do

- Do not add more semantic validation rules to `merge_quality.py`
- Do not write diagnostic text into the output document
- Do not increase complexity of retry profiles
- Do not conflate the publishable result with internal debug state

---

## Developer context

- **OS:** Windows 11, VSCode, `.bat` launcher scripts
- **Virtual env:** `.venv_restreamer`
- **Python:** explicit types, meaningful names, single responsibility
- **Multilingual:** project handles Russian, Ukrainian, English content
- **Author:** Arthur ([@TrustViking](https://github.com/TrustViking))
