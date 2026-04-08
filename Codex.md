# restreamer — Project Context for Claude Agent

## Session startup: auto-sync CLAUDE.md with actual project state

At the **start of every session**, before any other work, the agent MUST:

1. Read the current directory tree of `app/` (one level deep is enough, recurse where needed).
2. Compare it against the architecture table in the **Current architecture** section below.
3. If any packages are added, removed, or renamed — update the table to match reality.
4. Read `requirements.txt` and compare it against the **Key dependencies** list. Update if it differs.
5. Scan `app/llm/merges/merge_quality.py` briefly and update the **Current state** section if the situation has changed.
6. Update the **Priorities for next work sessions** list to reflect what is actually still pending vs. already done.
7. Do **not** touch the **Agent coding rules**, **Communication style**, or **Developer context** sections — they are project-independent and must not be auto-modified.

After syncing, save the updated CLAUDE.md (if anything changed) and proceed with the session task.

---

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
Bot entry point: `bot_main.py`
Main package: `app/`

| Package | Responsibility |
|---|---|
| `app/application` | App entry orchestration and runtime flow |
| `app/bootstrap` | Preflight, runtime context, startup health, logging |
| `app/config` | `.env` and runtime YAML configuration loading |
| `app/core` | Shared constants, models, domain utilities, exceptions |
| `app/google` | Google Drive/Sheets/Docs API clients |
| `app/ingest` | YouTube metadata extraction via `yt-dlp` |
| `app/llm` | Merge service, quality gate, parser, prompts, providers, usage tracking |
| `app/media` | Image normalization helpers |
| `app/modes` | Single-runner execution mode |
| `app/net` | HTTP client utilities |
| `app/observability` | Runtime analytics, summaries, health checks, usage tracking |
| `app/paths` | Name and path builders |
| `app/pipeline` | Core orchestration and slot processing |
| `app/planning` | Sheet loading, slot planning, merge rules, dedup |
| `app/publish` | Google Docs writer, post-LLM sanitation, Telegram payload rendering |
| `app/resources` | Prompt/lexicon/catalog text resources |
| `app/telegram` | Telegram bot API client |
| `app/telegram_bot` | Bot dispatcher, handlers, command routing |
| `app/tests` | Test suite |

**Key dependencies:** `yt-dlp`, `openai`, `google-api-python-client`, `google-genai`, `openpyxl`, `pillow`, `python-dotenv`, `aiogram`, `langdetect`, `pycountry`, `tzdata`, `pydantic`, `tenacity`

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

## LLM stack

- Provider abstraction in `app/llm/providers/` (`provider_base.py`, `provider_openai.py`)
- Model registry in `app/llm/models/`
- Prompts in `app/llm/prompts/`
- Rate limiting in `app/llm/llm_rate_limits.py`
- Usage tracking in `app/llm/llm_usage_tracker.py`
- `app/llm/__init__.py` uses lazy `__getattr__` re-exports — importing `app.llm` does NOT trigger the merge stack or `langdetect`
- Multi-step pipeline: generate → validate → retry → fallback

---

## Current state

### What works well
- Full batch pipeline from Sheets input to Docs + Telegram output
- Audit mode (nomerge vs merge comparison)
- Slot/language grouping
- Observability: run summaries, merge metrics, debug JSON artifacts
- Thumbnail/preview generation
- Telegram control bot (`bot_main.py`) for run/status commands
- Import decoupling: `app/llm/__init__.py` is lazy, does not eagerly load merge stack

### Completed refactors
- **Semantic checks removed** from `merge_quality.py` — only structural/format validation remains (32 lines, no subjective criteria)
- **Diagnostic annotations removed** from Google Docs output — doc layer is clean
- **`app/llm/__init__.py`** rewritten to lazy `__getattr__` — no eager `langdetect` import at package load
- **`setup_deploy.py`** preflight extended with `langdetect`, `pycountry`, `tzdata` import checks
- **`app/tests/test_import_cycle.py`** extended with 4 smoke-tests for isolated imports

---

## Priorities for next work sessions

1. **Simplify merge retry logic** — reduce retry profiles, remove targeted semantic retries
2. **Hard separation: doc = publishable result / JSON = full forensic data** — audit all paths where debug state could still reach the doc layer
3. No new heuristics or validation rules until the above is done

---

## What NOT to do

- Do not add semantic validation rules to `merge_quality.py`
- Do not write diagnostic text into the output document
- Do not increase complexity of retry profiles
- Do not conflate the publishable result with internal debug state
- Do not add eager imports to `app/llm/__init__.py` — keep lazy `__getattr__`

---

## Developer context

- **OS:** Windows 11, VSCode, `.bat` launcher scripts
- **Virtual env:** `.venv_restreamer` (Windows: `Scripts/`, not `bin/`)
- **Python:** explicit types, meaningful names, single responsibility
- **Multilingual:** project handles Russian, Ukrainian, English content
- **Author:** Arthur ([@TrustViking](https://github.com/TrustViking))

---

## Agent coding rules (MANDATORY for every edit)

These rules apply to ALL code you write or modify in this project.
Do not skip any of them. Do not ask whether to apply them — just apply them.

### Type annotations

- **Every** variable, parameter, and return value MUST have an explicit type annotation.
- Use `str`, `int`, `bool`, `float`, `None` for primitives.
- Use `list[str]`, `dict[str, int]`, `tuple[str, ...]` — lowercase generic syntax (Python 3.10+).
- Use `X | None` instead of `Optional[X]`.
- Return `None` explicitly: `def foo() -> None:`.
- Never leave a function signature without `-> ReturnType`.
- Dataclass fields must be annotated: `name: str = ""`.
- For complex types, import from `typing` only what you need (`TypeAlias`, `TypeVar`, `Protocol`).

```python
# CORRECT
def parse_slot(raw: str, fallback: str | None = None) -> SlotResult:
    merged_count: int = 0
    candidates: list[MergeCandidate] = []

# WRONG — never do this
def parse_slot(raw, fallback=None):
    merged_count = 0
    candidates = []
```

### Naming

- All names in English. No transliteration, no single-letter variables (except `i`, `j` in trivial loops).
- Functions: `verb_noun` — `build_merge_prompt`, `validate_slot_output`, `extract_video_metadata`.
- Boolean variables/params: `is_`, `has_`, `should_` prefix — `is_published`, `has_thumbnail`, `should_retry`.
- Constants: `UPPER_SNAKE_CASE` — `MAX_RETRY_COUNT`, `DEFAULT_LANGUAGE`.
- Private methods: single underscore prefix — `_compute_overlap`.
- No abbreviations unless universally known (`url`, `id`, `api`). Write `description`, not `desc`. Write `language`, not `lang`. Write `configuration`, not `cfg`.

### Function design

- **Single responsibility.** One function = one job. If you need an `and` to describe what it does, split it.
- **Max ~30 lines per function.** If longer, extract helpers.
- **No nested functions** unless absolutely necessary (closures for decorators are OK).
- **Guard clauses first** — early returns for invalid states, then the happy path.
- **No mutable default arguments** — use `None` + internal initialization.

```python
# CORRECT
def build_candidates(
    videos: list[VideoMeta],
    language: str,
    threshold: float | None = None,
) -> list[MergeCandidate]:
    if not videos:
        return []
    effective_threshold: float = threshold if threshold is not None else DEFAULT_THRESHOLD
    ...
```

### Error handling

- Catch specific exceptions, never bare `except:` or `except Exception:` without re-raising or logging.
- Every `except` block must either log the error or re-raise. Silent swallowing is forbidden.
- Use project-specific exceptions from `app/core/exceptions.py` when appropriate.

### Logging

- Use the project's logger, not `print()`.
- Logger names: hierarchical under `"restreamer"` — e.g. `logging.getLogger("restreamer.merge")`.
- DEBUG for internal state snapshots.
- INFO for operational milestones (slot processed, doc published, merge completed).
- WARNING for recoverable anomalies (fallback used, retry triggered).
- ERROR for failures that affect output.
- Include context: `logger.info("Merge completed slot=%s lang=%s candidates=%d", slot_id, language, count)`.

### Imports

- Group: stdlib → third-party → project. One blank line between groups.
- Absolute imports only: `from app.core.models import VideoMeta`, never relative.
- No wildcard imports (`from x import *`).
- No unused imports — clean them up before committing.
- `from __future__ import annotations` must be present in all modified files.

### Editing discipline

- **Do not reformat code you are not changing.** Touch only what the task requires.
- **Do not rename variables, functions, or files** outside the scope of the current task.
- **Do not add TODO/FIXME/HACK comments** — either fix the issue now or leave it alone.
- **Do not add new dependencies** without explicit approval.
- **Do not move files between packages** without explicit approval.
- **Preserve existing blank lines and section separators** in files you partially edit.
- When fixing a bug, explain the root cause in the commit message, not just what you changed.

### Testing

- If you modify logic, check whether existing tests in `app/tests/` cover that path.
- If you add a new public function, suggest a test. Do not create test files unless asked.
- Never modify test expectations to make a broken change pass.
- Run: `.venv_restreamer/Scripts/python.exe -m pytest app/tests/ --tb=short`

### Output separation (project-specific)

- **Google Docs output** = clean, publishable, human-readable. Zero diagnostic text.
- **debug/*.json** = full forensic data, rejected attempts, validation details.
- **logs/** = operational logging with timestamps and context.
- Never leak debug/diagnostic content into the publishable output layer.

---

## Communication style

- When presenting analysis or changes: reference exact file paths, function names, and line numbers.
- Do not offer a menu of options at the end — present your analysis, Arthur decides next steps.
- If a task is ambiguous, state your interpretation and proceed. Do not ask clarifying questions unless truly blocked.
- Respond in Russian if Arthur writes in Russian, in English if he writes in English.
