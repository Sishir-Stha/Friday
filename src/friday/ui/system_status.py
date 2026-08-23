from __future__ import annotations

import logging
from typing import Protocol

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from friday.system.monitor import SystemMetricsSnapshot

logger = logging.getLogger(__name__)

SYSTEM_REFRESH_INTERVAL_MS = 3000


class SystemMonitorLike(Protocol):
    def get_system_metrics(self) -> SystemMetricsSnapshot: ...


class SystemMetricsWorker(QObject):
    completed = Signal(object)
    failed = Signal()

    def __init__(self, monitor: SystemMonitorLike) -> None:
        super().__init__()
        self._monitor = monitor

    @Slot()
    def run(self) -> None:
        try:
            snapshot = self._monitor.get_system_metrics()
        except Exception:
            logger.exception("System metrics refresh failed")
            self.failed.emit()
        else:
            self.completed.emit(snapshot)


class SystemStatusWidget(QWidget):
    became_idle = Signal()

    def __init__(
        self,
        system_monitor: SystemMonitorLike,
        parent: QWidget | None = None,
        *,
        auto_start: bool = True,
    ) -> None:
        super().__init__(parent)
        self.system_monitor = system_monitor
        self._in_progress = False
        self._shutdown_requested = False
        self._thread: QThread | None = None
        self._worker: SystemMetricsWorker | None = None

        self.cpu_label = QLabel("CPU  —")
        self.ram_label = QLabel("RAM  —")
        self.gpu_label = QLabel("GPU  —")
        self.cpu_label.setObjectName("cpuStatus")
        self.ram_label.setObjectName("ramStatus")
        self.gpu_label.setObjectName("gpuStatus")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)
        layout.addWidget(self.cpu_label)
        layout.addWidget(self.ram_label)
        layout.addWidget(self.gpu_label)
        layout.addStretch(1)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(SYSTEM_REFRESH_INTERVAL_MS)
        self.refresh_timer.timeout.connect(self.refresh_now)
        if auto_start:
            self.refresh_timer.start()
            QTimer.singleShot(0, self.refresh_now)

    @property
    def refresh_in_progress(self) -> bool:
        return self._in_progress

    @property
    def has_active_worker(self) -> bool:
        return self._thread is not None

    @Slot()
    def refresh_now(self) -> None:
        if self._shutdown_requested or self._in_progress:
            return
        self._in_progress = True
        thread = QThread(self)
        worker = SystemMetricsWorker(self.system_monitor)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self.apply_snapshot)
        worker.failed.connect(self._refresh_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(object)
    def apply_snapshot(self, snapshot: SystemMetricsSnapshot) -> None:
        self.cpu_label.setText(f"CPU  {snapshot.cpu_percent:.0f}%")
        self.ram_label.setText(f"RAM  {snapshot.memory_percent:.0f}%")
        gpu_percent = None
        if snapshot.gpus:
            gpu_percent = snapshot.gpus[0].utilization_percent
        self.gpu_label.setText(
            "GPU  —" if gpu_percent is None else f"GPU  {gpu_percent:.0f}%"
        )

    @Slot()
    def _refresh_failed(self) -> None:
        pass

    @Slot()
    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._in_progress = False
        self.became_idle.emit()

    def begin_shutdown(self) -> None:
        self._shutdown_requested = True
        self.refresh_timer.stop()

    def stop(self) -> None:
        self.begin_shutdown()
