from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys

from app.paths._root import PROJECT_ROOT


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    entrypoint_path: Path
    runtime_config_path: Path
    runtime_config_example_path: Path
    templates_path: Path
    secrets_dir: Path
    secrets_env_path: Path
    oauth_token_path: Path
    oauth_credentials_path: Path
    service_account_path: Path
    bundled_config_path: Path
    bundled_templates_path: Path
    ytdlp_exe_path: Path
    cookies_file_path: Path
    deno_exe_path: Path
    state_dir: Path


@lru_cache(maxsize=1)
def get_project_paths() -> ProjectPaths:
    project_root: Path = PROJECT_ROOT
    entrypoint_path: Path = project_root / "bot_main.py"
    secrets_dir: Path = project_root / "secrets"
    frozen: bool = bool(getattr(sys, "frozen", False))

    if frozen:
        runtime_config_path: Path = project_root / "config" / "app_config.yaml"
        templates_path: Path = project_root / "config" / "templates.yaml"
        internal_root: Path = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        bundled_config_path: Path = (
            internal_root / "app" / "config" / "runtime" / "app_config.yaml"
        )
        bundled_templates_path: Path = (
            internal_root / "app" / "llm" / "prompts" / "templates.yaml"
        )
        runtime_config_example_path: Path = (
            internal_root / "app" / "config" / "runtime" / "app_config.example.yaml"
        )
    else:
        runtime_config_path = project_root / "app" / "config" / "runtime" / "app_config.yaml"
        runtime_config_example_path = (
            project_root / "app" / "config" / "runtime" / "app_config.example.yaml"
        )
        templates_path = project_root / "app" / "llm" / "prompts" / "templates.yaml"
        bundled_config_path = runtime_config_path
        bundled_templates_path = templates_path

    return ProjectPaths(
        project_root=project_root,
        entrypoint_path=entrypoint_path,
        runtime_config_path=runtime_config_path,
        runtime_config_example_path=runtime_config_example_path,
        templates_path=templates_path,
        secrets_dir=secrets_dir,
        secrets_env_path=secrets_dir / ".env",
        oauth_token_path=secrets_dir / "token.json",
        oauth_credentials_path=secrets_dir / "credentials.json",
        service_account_path=secrets_dir / "service_account.json",
        bundled_config_path=bundled_config_path,
        bundled_templates_path=bundled_templates_path,
        ytdlp_exe_path=project_root / "tools" / "yt-dlp.exe",
        cookies_file_path=secrets_dir / "cookies.txt",
        deno_exe_path=project_root / "tools" / "deno.exe",
        state_dir=project_root / "state",
    )
