from types import SimpleNamespace
from typing import Any

import pytest

import friday.system.app_launcher as launcher_module
from friday.system.app_launcher import (
    AppLauncher,
    AppLaunchResult,
    UnknownApplicationError,
    UnsupportedPlatformError,
)


def test_allowed_applications_are_sorted_immutable_aliases() -> None:
    applications = AppLauncher().list_allowed_apps()

    assert applications == ("calculator", "file_explorer", "notepad")
    assert isinstance(applications, tuple)


@pytest.mark.parametrize(
    ("alias", "expected_alias", "expected_executable"),
    [
        ("notepad", "notepad", "notepad.exe"),
        (" Calculator ", "calculator", "calc.exe"),
        ("FILE_EXPLORER", "file_explorer", "explorer.exe"),
    ],
)
def test_allowed_alias_resolves_to_fixed_executable(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
    expected_alias: str,
    expected_executable: str,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(launcher_module.platform, "system", lambda: "Windows")

    def fake_popen(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr(launcher_module.subprocess, "Popen", fake_popen)

    result = AppLauncher().open_app(alias)

    assert result == AppLaunchResult(app_name=expected_alias, pid=4321)
    assert calls == [([expected_executable], {"shell": False})]


@pytest.mark.parametrize(
    "app_name",
    [
        "unknown",
        "C:\\Windows\\System32\\cmd.exe",
        "notepad.exe",
        "notepad & calc",
        "powershell -Command Get-Process",
        "https://example.com",
        "file:///C:/Windows",
        "%COMSPEC%",
    ],
)
def test_unknown_or_command_looking_input_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    app_name: str,
) -> None:
    monkeypatch.setattr(launcher_module.platform, "system", lambda: "Windows")

    def unexpected_popen(*args: object, **kwargs: object) -> None:
        raise AssertionError("process must not be created")

    monkeypatch.setattr(launcher_module.subprocess, "Popen", unexpected_popen)

    with pytest.raises(UnknownApplicationError, match="is not allowed"):
        AppLauncher().open_app(app_name)


@pytest.mark.parametrize("app_name", ["", "   ", None, 1])
def test_invalid_app_name_is_rejected(app_name: Any) -> None:
    with pytest.raises(ValueError, match="app_name"):
        AppLauncher().open_app(app_name)


@pytest.mark.parametrize("platform_name", ["Linux", "Darwin"])
def test_non_windows_platform_is_rejected_without_process_creation(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
) -> None:
    monkeypatch.setattr(
        launcher_module.platform,
        "system",
        lambda: platform_name,
    )

    def unexpected_popen(*args: object, **kwargs: object) -> None:
        raise AssertionError("process must not be created")

    monkeypatch.setattr(launcher_module.subprocess, "Popen", unexpected_popen)

    with pytest.raises(UnsupportedPlatformError, match="only on Windows"):
        AppLauncher().open_app("notepad")


def test_missing_process_pid_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        launcher_module.subprocess,
        "Popen",
        lambda argv, **kwargs: object(),
    )

    result = AppLauncher().open_app("notepad")

    assert result.pid is None
