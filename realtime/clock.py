import threading
import time


class GlobalClock:
    """
    Monotonic realtime clock.

    Internal-only for Stage A.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._start_perf = 0.0
        self._offset_ms = 0.0
        self._running = False

    def start(self) -> None:
        with self._lock:
            self._start_perf = time.perf_counter()
            self._offset_ms = 0.0
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def now_ms(self) -> float:
        with self._lock:
            if not self._running:
                return 0.0
            elapsed = (time.perf_counter() - self._start_perf) * 1000.0
            return elapsed + self._offset_ms

    def set_offset_ms(self, offset_ms: float) -> None:
        with self._lock:
            self._offset_ms = float(offset_ms)

