from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from app.config.app_config_loader import load_google_auth_mode
from app.paths import ProjectPaths, get_project_paths


@dataclass(frozen=True)
class CheckResult:
    check_name: str
    status: str
    message: str


COLOR_GREEN: str = "\033[92m"
COLOR_YELLOW: str = "\033[93m"
COLOR_RED: str = "\033[91m"
COLOR_RESET: str = "\033[0m"

STATUS_PASS: str = "pass"
STATUS_WARNING: str = "warning"
STATUS_ERROR: str = "error"


def enable_ansi_on_windows() -> None:
    os.system("")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            return


def colorize_text(text: str, color: str) -> str:
    return f"{color}{text}{COLOR_RESET}"


def print_result(result: CheckResult) -> None:
    icon: str
    color: str
    if result.status == STATUS_PASS:
        icon = "✓"
        color = COLOR_GREEN
    elif result.status == STATUS_WARNING:
        icon = "⚠"
        color = COLOR_YELLOW
    else:
        icon = "✗"
        color = COLOR_RED
    formatted_message: str = colorize_text(
        f"{icon} {result.check_name}: {result.message}",
        color,
    )
    try:
        print(formatted_message)
    except UnicodeEncodeError:
        fallback_icon: str = "OK" if result.status == STATUS_PASS else "WARN"
        if result.status == STATUS_ERROR:
            fallback_icon = "ERR"
        print(colorize_text(f"{fallback_icon} {result.check_name}: {result.message}", color))


def check_python_version() -> CheckResult:
    version_info: tuple[int, int, int] = (
        int(sys.version_info.major),
        int(sys.version_info.minor),
        int(sys.version_info.micro),
    )
    version_label: str = f"{version_info[0]}.{version_info[1]}.{version_info[2]}"
    if (version_info[0], version_info[1]) < (3, 11):
        return CheckResult(
            check_name="Python version",
            status=STATUS_ERROR,
            message=f"{version_label} detected; Python 3.11+ is required.",
        )
    if (version_info[0], version_info[1]) < (3, 13):
        return CheckResult(
            check_name="Python version",
            status=STATUS_WARNING,
            message=f"{version_label} detected; Python 3.13+ is recommended.",
        )
    return CheckResult(
        check_name="Python version",
        status=STATUS_PASS,
        message=f"{version_label} detected.",
    )


def ensure_directory(path: Path) -> CheckResult:
    if path.exists() and not path.is_dir():
        return CheckResult(
            check_name=f"Directory {path.name}",
            status=STATUS_ERROR,
            message=f"{path} exists but is not a directory.",
        )
    if path.is_dir():
        return CheckResult(
            check_name=f"Directory {path.name}",
            status=STATUS_PASS,
            message="Exists.",
        )
    path.mkdir(parents=True, exist_ok=True)
    return CheckResult(
        check_name=f"Directory {path.name}",
        status=STATUS_WARNING,
        message="Missing directory was created automatically.",
    )


def ensure_required_directories(project_root: Path) -> list[CheckResult]:
    directory_names: list[str] = ["secrets", "logs", "docs"]
    if getattr(sys, "frozen", False):
        directory_names.append("config")
    results: list[CheckResult] = []
    directory_name: str
    for directory_name in directory_names:
        directory_path: Path = project_root / directory_name
        directory_result: CheckResult = ensure_directory(directory_path)
        results.append(directory_result)
    return results


def load_or_bootstrap_env(paths: ProjectPaths) -> tuple[list[CheckResult], dict[str, str]]:
    results: list[CheckResult] = []
    loaded_values: dict[str, str] = {}
    env_path: Path = paths.secrets_env_path
    example_path: Path = paths.project_root / "secrets" / ".env.example"
    dotenv_module_error: Exception | None = None
    dotenv_values_func: Any | None = None
    load_dotenv_func: Any | None = None

    try:
        dotenv_module: Any = importlib.import_module("dotenv")
        dotenv_values_func = getattr(dotenv_module, "dotenv_values")
        load_dotenv_func = getattr(dotenv_module, "load_dotenv")
    except Exception as error:
        dotenv_module_error = error

    if example_path.exists():
        if not env_path.exists():
            env_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(example_path, env_path)
            results.append(
                CheckResult(
                    check_name="Environment file",
                    status=STATUS_WARNING,
                    message=(
                        f"{env_path} was missing; copied template from {example_path}. "
                        "Fill required values manually."
                    ),
                )
            )
    elif not env_path.exists():
        results.append(
            CheckResult(
                check_name="Environment file",
                status=STATUS_ERROR,
                message=(
                    f"Missing {env_path} and {example_path}; cannot bootstrap environment file."
                ),
            )
        )
        return results, loaded_values

    if load_dotenv_func is None or dotenv_values_func is None:
        error_label: str = str(dotenv_module_error or "unknown error")
        results.append(
            CheckResult(
                check_name="Environment file",
                status=STATUS_ERROR,
                message=(
                    "python-dotenv is not available; cannot load secrets/.env "
                    f"({error_label})."
                ),
            )
        )
        return results, loaded_values

    load_dotenv_func(dotenv_path=env_path, override=False)
    parsed_values: Mapping[str, str | None] = dotenv_values_func(dotenv_path=env_path)
    loaded_values = {
        key: str(value)
        for key, value in parsed_values.items()
        if value is not None
    }
    if not any(item.check_name == "Environment file" for item in results):
        results.append(
            CheckResult(
                check_name="Environment file",
                status=STATUS_PASS,
                message=f"Loaded {env_path}.",
            )
        )
    return results, loaded_values


def validate_required_env_vars(env_values: Mapping[str, str]) -> list[CheckResult]:
    required_names: list[str] = [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "TELEGRAM_ADMIN_USER_IDS",
        "GPT_API_KEY",
        "GOOGLE_DRIVE_FOLDER_ID",
        "GOOGLE_SHEETS_ID",
    ]
    missing_names: list[str] = []
    variable_name: str
    for variable_name in required_names:
        value: str = str(env_values.get(variable_name, "") or "").strip()
        if not value:
            missing_names.append(variable_name)

    if missing_names:
        joined_names: str = ", ".join(missing_names)
        return [
            CheckResult(
                check_name="Required environment variables",
                status=STATUS_ERROR,
                message=f"Missing or empty: {joined_names}",
            )
        ]

    return [
        CheckResult(
            check_name="Required environment variables",
            status=STATUS_PASS,
            message="All required variables are present and non-empty.",
        )
    ]


def check_google_oauth_credentials(paths: ProjectPaths) -> CheckResult:
    credentials_path: Path = paths.oauth_credentials_path
    if credentials_path.exists() and credentials_path.is_file():
        return CheckResult(
            check_name="Google OAuth credentials",
            status=STATUS_PASS,
            message=f"Found {credentials_path}.",
        )
    return CheckResult(
        check_name="Google OAuth credentials",
        status=STATUS_ERROR,
        message=(
            f"Missing {credentials_path}. Download OAuth credentials from Google Cloud "
            "Console and place the file there."
        ),
    )


def check_google_service_account_credentials(paths: ProjectPaths) -> CheckResult:
    sa_path: Path = paths.service_account_path
    if sa_path.exists() and sa_path.is_file():
        return CheckResult(
            check_name="Google service account credentials",
            status=STATUS_PASS,
            message=f"Found {sa_path}.",
        )
    return CheckResult(
        check_name="Google service account credentials",
        status=STATUS_ERROR,
        message=(
            f"Missing {sa_path}. Place the service account JSON file there "
            "or set GOOGLE_SERVICE_ACCOUNT_PATH in .env."
        ),
    )


def ensure_runtime_config(paths: ProjectPaths) -> CheckResult:
    runtime_config_path: Path = paths.runtime_config_path
    runtime_example_path: Path = paths.runtime_config_example_path

    if runtime_config_path.exists() and runtime_config_path.is_file():
        return CheckResult(
            check_name="Runtime config",
            status=STATUS_PASS,
            message=f"Found {runtime_config_path}.",
        )

    if runtime_example_path.exists() and runtime_example_path.is_file():
        runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runtime_example_path, runtime_config_path)
        return CheckResult(
            check_name="Runtime config",
            status=STATUS_WARNING,
            message=(
                f"{runtime_config_path} was missing; copied from {runtime_example_path}."
            ),
        )

    return CheckResult(
        check_name="Runtime config",
        status=STATUS_ERROR,
        message=(
            f"Missing {runtime_config_path} and {runtime_example_path}; "
            "cannot bootstrap runtime config."
        ),
    )


def ensure_user_configs(paths: ProjectPaths) -> list[CheckResult]:
    """Copy bundled defaults to user-editable config paths when missing."""
    results: list[CheckResult] = []
    pairs: list[tuple[str, Path, Path]] = [
        ("app_config.yaml", paths.runtime_config_path, paths.bundled_config_path),
        ("templates.yaml", paths.templates_path, paths.bundled_templates_path),
    ]

    label: str
    user_path: Path
    bundled_path: Path
    for label, user_path, bundled_path in pairs:
        if user_path.exists():
            results.append(
                CheckResult(
                    check_name=f"Config {label}",
                    status=STATUS_PASS,
                    message=f"Found {user_path}.",
                )
            )
            continue

        if bundled_path.exists():
            user_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundled_path, user_path)
            results.append(
                CheckResult(
                    check_name=f"Config {label}",
                    status=STATUS_WARNING,
                    message=f"Copied bundled default to {user_path}.",
                )
            )
            continue

        results.append(
            CheckResult(
                check_name=f"Config {label}",
                status=STATUS_ERROR,
                message=f"Missing {user_path} and no bundled default at {bundled_path}.",
            )
        )

    return results


def validate_core_imports() -> CheckResult:
    required_modules: list[str] = [
        "aiogram",
        "openai",
        "dotenv",
        "yaml",
        "requests",
        "httpx",
        "google.auth",
        "google_auth_oauthlib",
        "googleapiclient",
        "PIL",
        "langdetect",
        "pycountry",
        "tzdata",
    ]
    failed_modules: list[str] = []

    module_name: str
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            failed_modules.append(module_name)

    if failed_modules:
        failed_label: str = ", ".join(failed_modules)
        return CheckResult(
            check_name="Core Python imports",
            status=STATUS_ERROR,
            message=f"Import failed for: {failed_label}",
        )

    return CheckResult(
        check_name="Core Python imports",
        status=STATUS_PASS,
        message="All required imports are available.",
    )


def check_telegram_api(token: str) -> CheckResult:
    safe_token: str = str(token or "").strip()
    if not safe_token:
        return CheckResult(
            check_name="Telegram API",
            status=STATUS_ERROR,
            message="TELEGRAM_BOT_TOKEN is missing; cannot run Telegram network check.",
        )

    try:
        httpx_module: Any = importlib.import_module("httpx")
    except Exception as error:
        return CheckResult(
            check_name="Telegram API",
            status=STATUS_ERROR,
            message=f"httpx is unavailable; cannot run network check ({error}).",
        )

    request_url: str = f"https://api.telegram.org/bot{safe_token}/getMe"
    timeout: Any = httpx_module.Timeout(timeout=20.0)

    try:
        with httpx_module.Client(timeout=timeout) as client:
            response: Any = client.get(request_url)
    except Exception as error:
        return CheckResult(
            check_name="Telegram API",
            status=STATUS_ERROR,
            message=f"Network error while calling Telegram getMe: {error}",
        )

    if response.status_code != 200:
        return CheckResult(
            check_name="Telegram API",
            status=STATUS_ERROR,
            message=f"Telegram getMe returned HTTP {response.status_code}.",
        )

    try:
        payload: object = response.json()
    except ValueError:
        return CheckResult(
            check_name="Telegram API",
            status=STATUS_ERROR,
            message="Telegram getMe returned non-JSON response.",
        )

    if not isinstance(payload, dict) or bool(payload.get("ok")) is not True:
        return CheckResult(
            check_name="Telegram API",
            status=STATUS_ERROR,
            message="Telegram getMe response has ok != true.",
        )

    return CheckResult(
        check_name="Telegram API",
        status=STATUS_PASS,
        message="Reachable and returned ok=true.",
    )


def check_openai_api(api_key: str) -> CheckResult:
    safe_api_key: str = str(api_key or "").strip()
    if not safe_api_key:
        return CheckResult(
            check_name="OpenAI API",
            status=STATUS_ERROR,
            message="GPT_API_KEY is missing; cannot run OpenAI network check.",
        )

    try:
        httpx_module: Any = importlib.import_module("httpx")
    except Exception as error:
        return CheckResult(
            check_name="OpenAI API",
            status=STATUS_ERROR,
            message=f"httpx is unavailable; cannot run network check ({error}).",
        )

    request_url: str = "https://api.openai.com/v1/models"
    headers: dict[str, str] = {"Authorization": f"Bearer {safe_api_key}"}
    timeout: Any = httpx_module.Timeout(timeout=20.0)

    try:
        with httpx_module.Client(timeout=timeout) as client:
            response: Any = client.get(request_url, headers=headers)
    except Exception as error:
        return CheckResult(
            check_name="OpenAI API",
            status=STATUS_ERROR,
            message=f"Network error while calling OpenAI models endpoint: {error}",
        )

    if response.status_code != 200:
        return CheckResult(
            check_name="OpenAI API",
            status=STATUS_ERROR,
            message=f"OpenAI models endpoint returned HTTP {response.status_code}.",
        )

    return CheckResult(
        check_name="OpenAI API",
        status=STATUS_PASS,
        message="Reachable and returned HTTP 200.",
    )


def print_summary(results: list[CheckResult]) -> int:
    passed_count: int = sum(1 for item in results if item.status == STATUS_PASS)
    warning_count: int = sum(1 for item in results if item.status == STATUS_WARNING)
    error_count: int = sum(1 for item in results if item.status == STATUS_ERROR)

    final_status: str = "READY" if error_count == 0 else "BLOCKED"
    final_color: str = COLOR_GREEN if error_count == 0 else COLOR_RED

    print("\nSummary")
    print(f"  Passed: {passed_count}")
    print(f"  Warnings: {warning_count}")
    print(f"  Errors: {error_count}")
    print(colorize_text(f"  Final status: {final_status}", final_color))

    return 0 if error_count == 0 else 1


def run_preflight(*, include_network_checks: bool) -> int:
    results: list[CheckResult] = []
    paths: ProjectPaths = get_project_paths()

    results.append(check_python_version())
    results.extend(ensure_required_directories(paths.project_root))
    if getattr(sys, "frozen", False):
        results.extend(ensure_user_configs(paths))

    env_results: list[CheckResult]
    env_values: dict[str, str]
    env_results, env_values = load_or_bootstrap_env(paths)
    results.extend(env_results)

    results.extend(validate_required_env_vars(env_values))
    resolved_auth_mode: str = load_google_auth_mode()
    if resolved_auth_mode == "service_account":
        results.append(check_google_service_account_credentials(paths))
    else:
        results.append(check_google_oauth_credentials(paths))
    results.append(ensure_runtime_config(paths))
    results.append(validate_core_imports())

    if include_network_checks:
        telegram_token: str = str(env_values.get("TELEGRAM_BOT_TOKEN", "") or "")
        openai_key: str = str(env_values.get("GPT_API_KEY", "") or "")
        results.append(check_telegram_api(telegram_token))
        results.append(check_openai_api(openai_key))
    else:
        results.append(
            CheckResult(
                check_name="Network checks",
                status=STATUS_WARNING,
                message="Skipped (run with --net to enable Telegram/OpenAI reachability checks).",
            )
        )

    print(f"Project root: {paths.project_root}")
    result_item: CheckResult
    for result_item in results:
        print_result(result_item)

    return print_summary(results)


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="deploy preflight checker"
    )
    parser.add_argument(
        "--net",
        action="store_true",
        dest="include_network_checks",
        help="Run optional network checks for Telegram and OpenAI APIs.",
    )
    return parser.parse_args()


def main() -> int:
    enable_ansi_on_windows()
    args: argparse.Namespace = parse_args()
    include_network_checks: bool = bool(args.include_network_checks)
    return run_preflight(include_network_checks=include_network_checks)


if __name__ == "__main__":
    raise SystemExit(main())
