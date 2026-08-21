from friday.system.monitor import GpuMetricsSnapshot, SystemMetricsSnapshot
from friday.ui.system_status import SYSTEM_REFRESH_INTERVAL_MS, SystemStatusWidget


def _snapshot(*, gpu_percent: float | None = 12.0) -> SystemMetricsSnapshot:
    gpus = ()
    if gpu_percent is not None:
        gpus = (
            GpuMetricsSnapshot(
                index=0,
                name="Test GPU",
                utilization_percent=gpu_percent,
                memory_used_mb=100,
                memory_total_mb=1000,
                temperature_c=50,
            ),
        )
    return SystemMetricsSnapshot(
        cpu_percent=24.4,
        memory_percent=58.2,
        memory_used_bytes=1,
        memory_available_bytes=1,
        memory_total_bytes=2,
        gpus=gpus,
    )


class FakeMonitor:
    def get_system_metrics(self) -> SystemMetricsSnapshot:
        return _snapshot()


def test_status_formatting_and_refresh_interval(qapp: object) -> None:
    widget = SystemStatusWidget(FakeMonitor(), auto_start=False)
    widget.apply_snapshot(_snapshot())

    assert widget.cpu_label.text() == "CPU  24%"
    assert widget.ram_label.text() == "RAM  58%"
    assert widget.gpu_label.text() == "GPU  12%"
    assert widget.refresh_timer.interval() == SYSTEM_REFRESH_INTERVAL_MS


def test_gpu_unavailable_and_failed_refresh_retain_values(qapp: object) -> None:
    widget = SystemStatusWidget(FakeMonitor(), auto_start=False)
    widget.apply_snapshot(_snapshot(gpu_percent=None))
    before = (
        widget.cpu_label.text(),
        widget.ram_label.text(),
        widget.gpu_label.text(),
    )

    widget._refresh_failed()

    assert widget.gpu_label.text() == "GPU  —"
    assert (
        widget.cpu_label.text(),
        widget.ram_label.text(),
        widget.gpu_label.text(),
    ) == before
    assert not widget.refresh_in_progress
