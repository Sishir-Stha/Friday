import platform
import subprocess
from dataclasses import dataclass
from types import MappingProxyType

_ALLOWED_APPLICATIONS = MappingProxyType(
    {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "file_explorer": "explorer.exe",
    }
)


@dataclass(frozen=True, slots=True)
class AppLaunchResult:
    app_name: str
    pid: int | None


class UnsupportedPlatformError(RuntimeError):
    """Raised when application launching is unavailable on this platform."""


class UnknownApplicationError(ValueError):
    """Raised when an application alias is not in Friday's fixed allowlist."""


class AppLauncher:
    def list_allowed_apps(self) -> tuple[str, ...]:
        return tuple(sorted(_ALLOWED_APPLICATIONS))

    def open_app(self, app_name: str) -> AppLaunchResult:
        normalized_name = _normalize_app_name(app_name)
        executable = _ALLOWED_APPLICATIONS.get(normalized_name)

        if executable is None:
            raise UnknownApplicationError(
                f"Application '{normalized_name}' is not allowed."
            )

        if platform.system() != "Windows":
            raise UnsupportedPlatformError(
                "Application launching is supported only on Windows."
            )

        process = subprocess.Popen(
            [executable],
            shell=False,
        )
        pid = getattr(process, "pid", None)

        return AppLaunchResult(
            app_name=normalized_name,
            pid=pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
        )


def _normalize_app_name(app_name: str) -> str:
    if not isinstance(app_name, str):
        raise ValueError("app_name must be a string")  # noqa: TRY004

    normalized = app_name.strip().lower()
    if not normalized:
        raise ValueError("app_name must not be empty")

    return normalized
