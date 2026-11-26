import dataclasses
import os
import shlex

import pytest

from zetaline.agent import config as cfg_mod
from zetaline.agent.config import (
    ENV_FILE_VAR,
    KEY_VAR,
    MODEL_VAR,
    AgentConfig,
    ConfigError,
    candidate_env_files,
    load_config,
)

SECRET = "AIzaFAKE-key_123"


def write_env(path, text, mode=0o600):
    path.write_text(text)
    os.chmod(path, mode)
    return path


def both(key=SECRET, model="some-model-1"):
    return f"{KEY_VAR}={key}\n{MODEL_VAR}={model}\n"


@pytest.fixture
def dirs(tmp_path):
    cwd, root = tmp_path / "cwd", tmp_path / "root"
    cwd.mkdir()
    root.mkdir()
    return cwd, root


def load(dirs, environ=None):
    cwd, root = dirs
    return load_config(environ=environ or {}, cwd=cwd, project_root=root)


def error_of(fn) -> str:
    with pytest.raises(ConfigError) as exc:
        fn()
    return str(exc.value)


# ---- finding and reading the file ---------------------------------------------------
def test_reads_key_and_model_from_cwd_file(dirs):
    path = write_env(dirs[0] / ".env", both())
    c = load(dirs)
    assert (c.api_key, c.model, c.source) == (SECRET, "some-model-1", str(path))


def test_falls_back_to_project_root_file(dirs):
    path = write_env(dirs[1] / ".env", both())
    assert load(dirs).source == str(path)


def test_cwd_file_is_preferred_over_project_root(dirs):
    write_env(dirs[0] / ".env", both(key="cwd-key"))
    write_env(dirs[1] / ".env", both(key="root-key"))
    assert load(dirs).api_key == "cwd-key"


def test_a_directory_named_dot_env_is_ignored(dirs):
    (dirs[0] / ".env").mkdir()
    path = write_env(dirs[1] / ".env", both())
    assert load(dirs).source == str(path)


def test_quotes_comments_and_whitespace_in_file(dirs):
    write_env(dirs[0] / ".env", f'# a comment\n{KEY_VAR}="abc123"\n{MODEL_VAR}= m-1 \n')
    c = load(dirs)
    assert (c.api_key, c.model) == ("abc123", "m-1")


def test_candidate_order_and_dedupe(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert candidate_env_files({}, a, b) == [a / ".env", b / ".env"]
    assert candidate_env_files({}, a, a) == [a / ".env"]


def test_candidates_use_only_the_explicit_file_when_set(tmp_path):
    assert candidate_env_files({ENV_FILE_VAR: "/x/y.env"}, tmp_path, tmp_path) == [
        cfg_mod.Path("/x/y.env")
    ]


def test_project_root_is_this_repository():
    assert (cfg_mod._PROJECT_ROOT / "pyproject.toml").is_file()


# ---- precedence -----------------------------------------------------------------------------
def test_environment_only_needs_no_file(dirs):
    c = load(dirs, {KEY_VAR: "k", MODEL_VAR: "m-2"})
    assert (c.api_key, c.model, c.source) == ("k", "m-2", "environment")


def test_environment_overrides_file_per_variable(dirs):
    path = write_env(dirs[0] / ".env", both())
    c = load(dirs, {MODEL_VAR: "env-model"})
    assert c.model == "env-model"
    assert c.api_key == SECRET and c.source == str(path)  # key still came from the file


def test_environment_key_overrides_file_key(dirs):
    write_env(dirs[0] / ".env", both())
    c = load(dirs, {KEY_VAR: "env-key"})
    assert (c.api_key, c.model, c.source) == ("env-key", "some-model-1", "environment")


def test_default_environ_is_the_process_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(KEY_VAR, "process-key")
    monkeypatch.setenv(MODEL_VAR, "process-model")
    monkeypatch.delenv(ENV_FILE_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    c = load_config()
    assert (c.api_key, c.model, c.source) == ("process-key", "process-model", "environment")


# ---- file permissions --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o666])
def test_loose_file_supplying_the_key_is_refused(dirs, mode):
    path = write_env(dirs[0] / ".env", both(), mode)
    message = error_of(lambda: load(dirs))
    assert "chmod 600" in message and str(path) in message
    assert SECRET not in message


@pytest.mark.parametrize("mode", [0o600, 0o400])
def test_private_modes_are_accepted(dirs, mode):
    write_env(dirs[0] / ".env", both(), mode)
    assert load(dirs).api_key == SECRET


def test_loose_file_is_fine_when_the_key_comes_from_the_environment(dirs):
    write_env(dirs[0] / ".env", f"{MODEL_VAR}=from-file\n", 0o644)
    c = load(dirs, {KEY_VAR: "k"})
    assert (c.model, c.source) == ("from-file", "environment")


def test_loose_file_is_refused_even_if_only_the_key_comes_from_it(dirs):
    write_env(dirs[0] / ".env", f"{KEY_VAR}={SECRET}\n", 0o644)
    assert "chmod 600" in error_of(lambda: load(dirs, {MODEL_VAR: "m"}))


def test_chmod_hint_is_shell_quoted_for_paths_with_spaces(tmp_path):
    spaced = tmp_path / "has space"
    spaced.mkdir()
    path = write_env(spaced / ".env", both(), 0o644)
    message = error_of(lambda: load_config(environ={}, cwd=spaced, project_root=spaced))
    assert shlex.quote(str(path)) in message and "'" in message


# ---- ZETALINE_ENV_FILE -----------------------------------------------------------------------------
def test_explicit_file_is_the_only_file_used(dirs, tmp_path):
    write_env(dirs[0] / ".env", both(key="cwd-key"))
    explicit = write_env(tmp_path / "other.env", both(key="explicit-key", model="m-3"))
    c = load(dirs, {ENV_FILE_VAR: str(explicit)})
    assert (c.api_key, c.model, c.source) == ("explicit-key", "m-3", str(explicit))


def test_explicit_missing_file_is_an_error_not_a_fallback(dirs, tmp_path):
    write_env(dirs[0] / ".env", both())  # would be found if we fell back
    message = error_of(lambda: load(dirs, {ENV_FILE_VAR: str(tmp_path / "nope.env")}))
    assert "missing file" in message and ENV_FILE_VAR in message


def test_explicit_file_expands_the_home_directory(dirs, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_env(tmp_path / "e.env", both(key="home-key"))
    assert load(dirs, {ENV_FILE_VAR: "~/e.env"}).api_key == "home-key"


# ---- missing and blank values -------------------------------------------------------------------------
def test_missing_key_is_reported_first_and_without_values(dirs):
    message = error_of(lambda: load(dirs))
    assert KEY_VAR in message and MODEL_VAR not in message


def test_missing_model(dirs):
    message = error_of(lambda: load(dirs, {KEY_VAR: SECRET}))
    assert MODEL_VAR in message and SECRET not in message


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_environment_values_count_as_unset(dirs, blank):
    assert KEY_VAR in error_of(lambda: load(dirs, {KEY_VAR: blank, MODEL_VAR: "m"}))
    assert MODEL_VAR in error_of(lambda: load(dirs, {KEY_VAR: "k", MODEL_VAR: blank}))


def test_blank_and_valueless_lines_in_file_count_as_unset(dirs):
    write_env(dirs[0] / ".env", f"{KEY_VAR}\n{MODEL_VAR}=   \n")
    assert KEY_VAR in error_of(lambda: load(dirs))


def test_blank_environment_value_falls_back_to_the_file(dirs):
    write_env(dirs[0] / ".env", both())
    assert load(dirs, {KEY_VAR: "  "}).api_key == SECRET


def test_key_with_whitespace_inside_is_refused_without_echoing_it(dirs):
    message = error_of(lambda: load(dirs, {KEY_VAR: "part1 SECRETPART", MODEL_VAR: "m"}))
    assert "SECRETPART" not in message and KEY_VAR in message


# ---- AgentConfig ------------------------------------------------------------------------------------------
def test_agent_config_basics():
    c = AgentConfig("key123", "gemini-x")
    assert c.source == "environment"
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.model = "other"


def test_key_is_hidden_from_repr_and_str():
    c = AgentConfig(SECRET, "m")
    assert SECRET not in repr(c) and SECRET not in str(c)
    assert "model='m'" in repr(c)


def test_redacted_shows_the_length_only():
    shown = AgentConfig(SECRET, "m").redacted()
    assert str(len(SECRET)) in shown and SECRET not in shown and "AIza" not in shown


@pytest.mark.parametrize("key", ["", "a b", "a\tb", "a\nb", "key\n"])
def test_invalid_keys_are_refused(key):
    with pytest.raises(ConfigError, match=KEY_VAR):
        AgentConfig(key, "m")


@pytest.mark.parametrize(
    "model", ["", "bad model!", "-leading", ".leading", "x" * 101, "m\n", "m/odel", " m"]
)
def test_invalid_models_are_refused(model):
    with pytest.raises(ConfigError, match=MODEL_VAR):
        AgentConfig("k", model)


@pytest.mark.parametrize("model", ["a", "gemini-2.5-flash", "Model_1.2-b", "x" * 100])
def test_valid_models_are_accepted(model):
    assert AgentConfig("k", model).model == model
