from dataclasses import asdict

import pytest

from friday.system.monitor import SystemMonitor

pytestmark = pytest.mark.integration


def test_real_system_monitor_returns_stable_snapshot_fields() -> None:
    monitor = SystemMonitor()

    info = monitor.get_system_info()
    metrics = monitor.get_system_metrics()

    print("\n===== FRIDAY SYSTEM INFO =====")
    print(asdict(info))
    print("\n===== FRIDAY SYSTEM METRICS =====")
    print(asdict(metrics))

    assert info.hostname.strip()
    assert info.logical_cpu_count is None or info.logical_cpu_count > 0
    assert info.physical_cpu_count is None or info.physical_cpu_count > 0
    assert info.total_memory_bytes > 0

    assert 0 <= metrics.cpu_percent <= 100
    assert 0 <= metrics.memory_percent <= 100
    assert metrics.memory_used_bytes >= 0
    assert metrics.memory_available_bytes >= 0
    assert metrics.memory_total_bytes > 0

    for gpu in metrics.gpus:
        assert gpu.index >= 0
        assert gpu.name.strip()
        assert gpu.utilization_percent is None or gpu.utilization_percent >= 0
        assert gpu.memory_used_mb is None or gpu.memory_used_mb >= 0
        assert gpu.memory_total_mb is None or gpu.memory_total_mb > 0
        assert gpu.temperature_c is None or gpu.temperature_c >= 0
