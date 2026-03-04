from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


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


@lru_cache(maxsize=1)
def get_project_paths() -> ProjectPaths:
    project_root: Path = Path(__file__).resolve().parents[2]
    entrypoint_path: Path = project_root / "restreamer.py"
    secrets_dir: Path = project_root / "secrets"
    return ProjectPaths(
        project_root=project_root,
        entrypoint_path=entrypoint_path,
        runtime_config_path=project_root / "app" / "config" / "runtime" / "app_config.yaml",
        runtime_config_example_path=project_root
        / "app"
        / "config"
        / "runtime"
        / "app_config.example.yaml",
        templates_path=project_root / "app" / "llm" / "prompts" / "templates.yaml",
        secrets_dir=secrets_dir,
        secrets_env_path=secrets_dir / ".env",
        oauth_token_path=secrets_dir / "token.json",
        oauth_credentials_path=secrets_dir / "credentials.json",
    )
