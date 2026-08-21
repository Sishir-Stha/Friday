import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

import friday.system.monitor as monitor_module
from friday.system.monitor import SystemMonitor


def memory_snapshot() -> SimpleNamespace:
    return SimpleNamespace(
        percent=62.5,
        used=8_000,
        available=4_000,
        total=12_000,
    )


def test_system_info_uses_local_platform_and_psutil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(monitor_module.socket, "gethostname", lambda: "friday-pc")
    monkeypatch.setattr(monitor_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(monitor_module.platform, "release", lambda: "11")
    monkeypatch.setattr(monitor_module.platform, "version", lambda: "10.0.1")
    monkeypatch.setattr(monitor_module.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(monitor_module.platform, "processor", lambda: "Test CPU")
    monkeypatch.setattr(
        monitor_module.psutil,
        "cpu_count",
        lambda *, logical: 8 if logical else 4,
    )
    monkeypatch.setattr(
        monitor_module.psutil,
        "virtual_memory",
        memory_snapshot,
    )

    snapshot = SystemMonitor().get_system_info()

    assert snapshot.hostname == "friday-pc"
    assert snapshot.os_name == "Windows"
    assert snapshot.os_release == "11"
    assert snapshot.os_version == "10.0.1"
    assert snapshot.architecture == "AMD64"
    assert snapshot.processor == "Test CPU"
    assert snapshot.physical_cpu_count == 4
    assert snapshot.logical_cpu_count == 8
    assert snapshot.total_memory_bytes == 12_000


def test_system_metrics_uses_bounded_cpu_sample_and_ram_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intervals: list[float] = []
    monkeypatch.setattr(
        monitor_module.psutil,
        "cpu_percent",
        lambda *, interval: intervals.append(interval) or 37.5,
    )
    monkeypatch.setattr(
        monitor_module.psutil,
        "virtual_memory",
        memory_snapshot,
    )
    monkeypatch.setattr(monitor_module.shutil, "which", lambda name: None)

    snapshot = SystemMonitor().get_system_metrics()

    assert intervals == [0.1]
    assert snapshot.cpu_percent == 37.5
    assert snapshot.memory_percent == 62.5
    assert snapshot.memory_used_bytes == 8_000
    assert snapshot.memory_available_bytes == 4_000
    assert snapshot.memory_total_bytes == 12_000
    assert snapshot.gpus == ()


def configure_gpu_query(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: str,
    returncode: int = 0,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        monitor_module.shutil,
        "which",
        lambda name: "C:\\NVIDIA\\nvidia-smi.exe",
    )

    def fake_run(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append({"argv": argv, **kwargs})
        return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)

    monkeypatch.setattr(monitor_module.subprocess, "run", fake_run)
    return calls


def test_gpu_query_supports_multiple_rows_and_fixed_safe_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = configure_gpu_query(
        monkeypatch,
        stdout=(
            "0, NVIDIA GeForce GTX 950M, 42, 1200, 4096, 65\n"
            "1, NVIDIA Test GPU, 10.5, 200, 8192, 44\n"
        ),
    )

    gpus = SystemMonitor()._get_gpu_metrics()

    assert len(gpus) == 2
    assert gpus[0].index == 0
    assert gpus[0].name == "NVIDIA GeForce GTX 950M"
    assert gpus[0].utilization_percent == 42.0
    assert gpus[0].memory_used_mb == 1200.0
    assert gpus[0].memory_total_mb == 4096.0
    assert gpus[0].temperature_c == 65.0
    assert gpus[1].index == 1
    assert calls[0]["argv"] == [
        "C:\\NVIDIA\\nvidia-smi.exe",
        (
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu"
        ),
        "--format=csv,noheader,nounits",
    ]
    assert calls[0]["capture_output"] is True
    assert calls[0]["text"] is True
    assert calls[0]["timeout"] == 3
    assert calls[0]["check"] is False
    assert calls[0]["shell"] is False


def test_gpu_na_numeric_values_become_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_gpu_query(
        monkeypatch,
        stdout="0, NVIDIA GPU, N/A, N/A, 4096, N/A\n",
    )

    gpu = SystemMonitor()._get_gpu_metrics()[0]

    assert gpu.utilization_percent is None
    assert gpu.memory_used_mb is None
    assert gpu.memory_total_mb == 4096.0
    assert gpu.temperature_c is None


def test_missing_nvidia_smi_returns_no_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(monitor_module.shutil, "which", lambda name: None)

    def unexpected_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(monitor_module.subprocess, "run", unexpected_run)

    assert SystemMonitor()._get_gpu_metrics() == ()


def test_gpu_timeout_returns_no_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        monitor_module.shutil,
        "which",
        lambda name: "nvidia-smi.exe",
    )

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=3)

    monkeypatch.setattr(monitor_module.subprocess, "run", timeout)

    assert SystemMonitor()._get_gpu_metrics() == ()


def test_gpu_os_error_returns_no_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        monitor_module.shutil,
        "which",
        lambda name: "nvidia-smi.exe",
    )
    monkeypatch.setattr(
        monitor_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unavailable")),
    )

    assert SystemMonitor()._get_gpu_metrics() == ()


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [("", 0), ("   \n", 0), ("query failed", 1)],
)
def test_unavailable_gpu_command_result_returns_no_gpus(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    returncode: int,
) -> None:
    configure_gpu_query(
        monkeypatch,
        stdout=stdout,
        returncode=returncode,
    )

    assert SystemMonitor()._get_gpu_metrics() == ()


@pytest.mark.parametrize(
    "malformed_row",
    [
        "not,csv",
        "index, GPU, 10, 20, 30, 40",
        "-1, GPU, 10, 20, 30, 40",
        "0, , 10, 20, 30, 40",
        "0, GPU, broken, 20, 30, 40",
    ],
)
def test_malformed_gpu_rows_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
    malformed_row: str,
) -> None:
    configure_gpu_query(monkeypatch, stdout=f"{malformed_row}\n")

    assert SystemMonitor()._get_gpu_metrics() == ()


def test_monitor_has_no_background_state() -> None:
    first = SystemMonitor()
    second = SystemMonitor()

    assert first is not second
    assert first.__dict__ == {}
    assert second.__dict__ == {}
