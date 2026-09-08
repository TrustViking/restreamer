Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Broadcaster packaging rules

- The application icon is `ico_code.ico`. Keep `broadcaster.spec`,
  `build_broadcaster_exe.bat`, `installer.iss`, and `dist_layout.md` in sync.
- Public examples must remain trackable in git: `.env.example`,
  `secrets/.env.example`, `app/config/runtime/app_config.example.yaml`.
- Real secrets must never be committed: `.env`, `token.json`, `credentials.json`,
  `service_account.json`, `cookies.txt`.
- The hidden duplicate `app/config/runtime/.app_config.example.yaml` is ignored.
- `app/config/runtime/app_config.yaml` is the operator's personal config
  (form URL, contacts) and is NOT tracked in git. Only
  `app_config.example.yaml` is tracked. `broadcaster.spec` still bundles
  `app_config.yaml`, so a fresh clone must copy the example to that name
  before building. Never re-add `app_config.yaml` to git.
- Single-instance enforcement is at the **process level** via a lock file
  `state/broadcaster.lock` (managed by `app/runtime/single_instance.py`).
  Acquired in `bot_main.py:main()` BEFORE logging setup. On conflict the
  process writes a Russian operator message to stderr AND appends one
  `lock_rejected ...` line to `logs/bot_startup.log`, then exits with
  code 1. Stale locks (PID dead) are overwritten via atomic recovery —
  `unlink` then retry `O_EXCL` create exactly once. If the recovery
  `O_EXCL` also loses, the policy is **fail-closed**: raise
  `AnotherInstanceRunning` rather than overwrite. Never write to an
  existing lock file via `write_text` — only through fresh `O_EXCL`
  creates. Released in `finally` on every exit path including
  BaseException. Successful acquire and release are mirrored to
  `logs/bot_startup.log` so all startup-phase events are visible in the
  same shippable `logs/` directory (NOT in `state/`, which is runtime
  data and never shipped for analysis).
- Telegram-side conflict handling (`TelegramConflictError`, getUpdates probe)
  was tried in a previous round and **proven insufficient** in production logs
  (20260512_15* — three bots passed probe, three pipelines ran, 12 duplicate
  publications). Do not reintroduce the probe. The process-level lock is the
  source of truth.
- Within-process duplicate triggers (one user spam-clicking the run button
  inside the same bot process) are handled by `_pipeline_lock: threading.Lock`
  in `bot_handlers_info.py` — leave that mechanism alone.
- Run-status resolution in `_resolve_run_status` (analytics_formatters.py)
  considers `fallback_merge_blocks` and `partial_merge_artifacts` as
  `partial` signals. Do not let `run_final_summary status=success` slip
  through when `audit_done overall_status=partial`.
- Portable build target is `dist\broadcaster\`. It contains both real config
  and example config so the same folder can be redistributed minus secrets.
- Installer build is two-track:
  - `build_release.bat` strips real secrets AND replaces `app_config.yaml`
    with `app_config.example.yaml` in the staging folder before invoking
    ISCC. Output filename is `broadcaster-setup-<version>.exe`. Safe for
    GitHub Releases.
  - `build_local.bat` keeps real secrets and real `app_config.yaml`.
    Output filename is `broadcaster-setup-local-<version>.exe` to prevent
    overwriting a release build sitting in `dist\installer\`.
    For personal use only.
- The installer (`installer.iss`) writes only to `{app}` and the standard
  Inno uninstall entry. It must not add custom registry keys.
- Do not use broad ignore rules like `*.bat` or `*.ico` — they hide
  intentional project files.
