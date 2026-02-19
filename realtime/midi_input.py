import threading
import time
from typing import Callable, Optional

import mido


MidiCallback = Callable[[mido.Message, float], None]


class MIDIIO:
    """
    Poll-based MIDI input/output adapter for realtime runtime.
    """

    def __init__(
        self,
        input_port_name: Optional[str] = None,
        output_port_name: Optional[str] = None,
        poll_interval_ms: float = 1.0,
        passthrough: bool = False,
    ):
        self.input_port_name = input_port_name
        self.output_port_name = output_port_name
        self.poll_interval_ms = poll_interval_ms
        self.passthrough = passthrough

        self._input_port = None
        self._output_port = None
        self._callback: Optional[MidiCallback] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @staticmethod
    def list_input_ports():
        return mido.get_input_names()

    @staticmethod
    def list_output_ports():
        return mido.get_output_names()

    def _pick_port_name(self, names, preferred: Optional[str], direction: str) -> str:
        if preferred:
            if preferred not in names:
                raise ValueError(
                    f"Requested {direction} port '{preferred}' not found. Available: {names}"
                )
            return preferred
        if not names:
            raise ValueError(f"No MIDI {direction} ports available.")
        return names[0]

    def open(self) -> None:
        input_names = self.list_input_ports()
        output_names = self.list_output_ports()
        input_name = self._pick_port_name(input_names, self.input_port_name, "input")
        output_name = self._pick_port_name(output_names, self.output_port_name, "output")

        self._input_port = mido.open_input(input_name)
        self._output_port = mido.open_output(output_name)

        self.input_port_name = input_name
        self.output_port_name = output_name

    def set_callback(self, callback: MidiCallback) -> None:
        self._callback = callback

    def send(self, msg: mido.Message) -> None:
        if self._output_port is not None:
            self._output_port.send(msg)

    def start(self, clock_now_ms: Callable[[], float]) -> None:
        if self._input_port is None or self._output_port is None:
            self.open()

        if self._callback is None:
            raise RuntimeError("MIDI callback is not set.")

        self._stop_event.clear()

        def _run():
            while not self._stop_event.is_set():
                for msg in self._input_port.iter_pending():
                    t_ms = clock_now_ms()
                    self._callback(msg, t_ms)
                    if self.passthrough and msg.type in ("note_on", "note_off"):
                        self.send(msg)
                time.sleep(self.poll_interval_ms / 1000.0)

        self._thread = threading.Thread(target=_run, name="midi-input-thread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        if self._input_port is not None:
            self._input_port.close()
            self._input_port = None
        if self._output_port is not None:
            self._output_port.close()
            self._output_port = None

