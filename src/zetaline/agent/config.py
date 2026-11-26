"""Agent configuration: API key and model, from the environment or a .env file.

The key is never printed: AgentConfig hides it from repr/str, and error messages name
variables, never values. A .env file that supplies the key must be private (no group or
world access). Environment variables take precedence over the file, per variable.
"""
from __future__ import annotations

import os
import re
import shlex
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

KEY_VAR = "GEMINI_API_KEY"
MODEL_VAR = "ZETALINE_AGENT_MODEL"
ENV_FILE_VAR = "ZETALINE_ENV_FILE"

_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ConfigError(Exception):
    """Configuration is missing or invalid. Messages never contain the API key."""


@dataclass(frozen=True, slots=True)
class AgentConfig:
    api_key: str = field(repr=False)
    model: str
    source: str = "environment"  # where the key came from: "environment" or a file path

    def __post_init__(self) -> None:
        if not self.api_key or any(ch.isspace() for ch in self.api_key):
            raise ConfigError(f"{KEY_VAR} is empty or contains whitespace")
        if not _MODEL_RE.fullmatch(self.model):
            raise ConfigError(f"{MODEL_VAR} is not a valid model name: {self.model!r}")

    def redacted(self) -> str:
        return f"<redacted, {len(self.api_key)} chars>"


def candidate_env_files(environ: Mapping[str, str], cwd: Path, project_root: Path) -> list[Path]:
    explicit = environ.get(ENV_FILE_VAR)
    if explicit:
        return [Path(explicit).expanduser()]
    found: list[Path] = []
    for base in (cwd, project_root):
        path = base / ".env"
        if path not in found:
            found.append(path)
    return found


def _read_env_file(
    environ: Mapping[str, str], cwd: Path, project_root: Path
) -> tuple[Path | None, dict[str, str]]:
    candidates = candidate_env_files(environ, cwd, project_root)
    for candidate in candidates:
        if candidate.is_file():
            raw = dotenv_values(candidate)
            return candidate, {k: v.strip() for k, v in raw.items() if v and v.strip()}
    if environ.get(ENV_FILE_VAR):
        raise ConfigError(f"{ENV_FILE_VAR} points to a missing file: {candidates[0]}")
    return None, {}


def _require_private(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ConfigError(
            f"{path} is accessible to other users (mode {mode:04o}); "
            f"run: chmod 600 {shlex.quote(str(path))}"
        )


def load_config(
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    project_root: Path | str | None = None,
) -> AgentConfig:
    env = os.environ if environ is None else environ
    here = Path.cwd() if cwd is None else Path(cwd)
    root = _PROJECT_ROOT if project_root is None else Path(project_root)
    path, file_values = _read_env_file(env, here, root)

    def pick(name: str) -> tuple[str, str]:
        value = (env.get(name) or "").strip()
        if value:
            return value, "environment"
        value = file_values.get(name, "")
        if value and path is not None:
            return value, str(path)
        return "", ""

    key, key_source = pick(KEY_VAR)
    model, _ = pick(MODEL_VAR)
    if not key:
        raise ConfigError(
            f"{KEY_VAR} is not set. Put it in a git-ignored .env file (see .env.example) "
            "or in the environment."
        )
    if not model:
        raise ConfigError(
            f"{MODEL_VAR} is not set. Choose a model from your AI Studio list and set it in .env."
        )
    if path is not None and key_source == str(path):
        _require_private(path)
    return AgentConfig(api_key=key, model=model, source=key_source)
