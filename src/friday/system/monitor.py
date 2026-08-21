import csv
import platform
import shutil
import socket
import subprocess
from dataclasses import dataclass

import psutil

_CPU_SAMPLE_INTERVAL_SECONDS = 0.1
_NVIDIA_SMI_TIMEOUT_SECONDS = 3
_NVIDIA_SMI_QUERY = (
    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
    "temperature.gpu"
)


@dataclass(frozen=True, slots=True)
class SystemInfoSnapshot:
    hostname: str
    os_name: str
    os_release: str
    os_version: str
    architecture: str
    processor: str
    physical_cpu_count: int | None
    logical_cpu_count: int | None
    total_memory_bytes: int


@dataclass(frozen=True, slots=True)
class GpuMetricsSnapshot:
    index: int
    name: str
    utilization_percent: float | None
    memory_used_mb: float | None
    memory_total_mb: float | None
    temperature_c: float | None


@dataclass(frozen=True, slots=True)
class SystemMetricsSnapshot:
    cpu_percent: float
    memory_percent: float
    memory_used_bytes: int
    memory_available_bytes: int
    memory_total_bytes: int
    gpus: tuple[GpuMetricsSnapshot, ...]


class SystemMonitor:
    def get_system_info(self) -> SystemInfoSnapshot:
        memory = psutil.virtual_memory()
        return SystemInfoSnapshot(
            hostname=socket.gethostname(),
            os_name=platform.system(),
            os_release=platform.release(),
            os_version=platform.version(),
            architecture=platform.machine(),
            processor=platform.processor(),
            physical_cpu_count=psutil.cpu_count(logical=False),
            logical_cpu_count=psutil.cpu_count(logical=True),
            total_memory_bytes=int(memory.total),
        )

    def get_system_metrics(self) -> SystemMetricsSnapshot:
        cpu_percent = psutil.cpu_percent(
            interval=_CPU_SAMPLE_INTERVAL_SECONDS,
        )
        memory = psutil.virtual_memory()

        return SystemMetricsSnapshot(
            cpu_percent=float(cpu_percent),
            memory_percent=float(memory.percent),
            memory_used_bytes=int(memory.used),
            memory_available_bytes=int(memory.available),
            memory_total_bytes=int(memory.total),
            gpus=self._get_gpu_metrics(),
        )

    @staticmethod
    def _get_gpu_metrics() -> tuple[GpuMetricsSnapshot, ...]:
        nvidia_smi_path = shutil.which("nvidia-smi")
        if nvidia_smi_path is None:
            return ()

        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            completed = subprocess.run(
                [
                    nvidia_smi_path,
                    _NVIDIA_SMI_QUERY,
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
                check=False,
                shell=False,
                creationflags=creation_flags,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ()

        if completed.returncode != 0 or not completed.stdout.strip():
            return ()

        return _parse_gpu_metrics(completed.stdout)


def _parse_gpu_metrics(output: str) -> tuple[GpuMetricsSnapshot, ...]:
    snapshots: list[GpuMetricsSnapshot] = []

    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            continue

        values = [value.strip() for value in row]
        try:
            index = int(values[0])
            if index < 0 or not values[1]:
                continue
            snapshot = GpuMetricsSnapshot(
                index=index,
                name=values[1],
                utilization_percent=_parse_optional_float(values[2]),
                memory_used_mb=_parse_optional_float(values[3]),
                memory_total_mb=_parse_optional_float(values[4]),
                temperature_c=_parse_optional_float(values[5]),
            )
        except ValueError:
            continue

        snapshots.append(snapshot)

    return tuple(snapshots)


def _parse_optional_float(value: str) -> float | None:
    if value.upper() == "N/A":
        return None
    return float(value)
