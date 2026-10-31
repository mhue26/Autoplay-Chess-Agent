"""Background worker threads for training and benchmarking with UI queue messaging."""

from __future__ import annotations

import inspect
import queue
import threading
import time
import traceback
from typing import Any, Callable, Dict, Optional

from rl_chess.config import DEFAULT_CONFIG_PATH
from rl_chess.benchmark import run_benchmark
from rl_chess.train.loop import run_training


class TrainingWorker:
    def __init__(self, event_queue: queue.Queue):
        self.event_queue = event_queue
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_active = False
        self.pace_delay_sec: float = 0.0

    def set_pace_delay(self, delay: float) -> None:
        self.pace_delay_sec = max(0.0, float(delay))

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def request_stop(self) -> None:
        if self.is_running():
            self._stop_event.set()
            self.event_queue.put({"type": "log", "text": "Stop requested... waiting for current step/game to finish."})

    def start(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        generations: int = 1,
        pace_delay_sec: float = 0.0,
    ) -> bool:
        if self.is_running():
            return False

        self.pace_delay_sec = max(0.0, float(pace_delay_sec))
        self._stop_event.clear()
        self.is_active = True

        def run():
            self.event_queue.put({"type": "train_status", "status": "running"})
            try:
                def on_log(msg: str):
                    self.event_queue.put({"type": "log", "target": "train", "text": msg})

                def on_progress(data: Dict[str, Any]):
                    self.event_queue.put({"type": "train_progress", "data": data})

                def on_event(event: Dict[str, Any]):
                    self.event_queue.put({"type": "train_viz", "data": event})
                    if self.pace_delay_sec > 0 and event.get("type") == "move":
                        time.sleep(self.pace_delay_sec)

                run_training(
                    config_path=config_path,
                    generations=generations,
                    on_log=on_log,
                    on_progress=on_progress,
                    on_event=on_event,
                    stop_event=self._stop_event,
                )
                if self._stop_event.is_set():
                    self.event_queue.put({"type": "train_status", "status": "stopped"})
                else:
                    self.event_queue.put({"type": "train_status", "status": "completed"})
            except Exception as e:
                err = f"Training Error: {e}\n{traceback.format_exc()}"
                self.event_queue.put({"type": "log", "target": "train", "text": err})
                self.event_queue.put({"type": "train_status", "status": "error", "error": str(e)})
            finally:
                self.is_active = False

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return True


class BenchmarkWorker:
    def __init__(self, event_queue: queue.Queue):
        self.event_queue = event_queue
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_active = False

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def request_stop(self) -> None:
        if self.is_running():
            self._stop_event.set()
            self.event_queue.put({"type": "log", "text": "Benchmark stop requested... finalizing current game."})

    def start(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        checkpoint_target: str = "",
        games_per_opponent: int = 10,
    ) -> bool:
        if self.is_running():
            return False

        self._stop_event.clear()
        self.is_active = True

        def run():
            self.event_queue.put({"type": "bench_status", "status": "running"})
            try:
                def on_log(msg: str):
                    self.event_queue.put({"type": "log", "target": "bench", "text": msg})

                def on_progress(data: Dict[str, Any]):
                    self.event_queue.put({"type": "bench_progress", "data": data})

                def on_result(data: Dict[str, Any]):
                    self.event_queue.put({"type": "bench_result", "data": data})

                run_benchmark(
                    config_path=config_path,
                    checkpoint_target=checkpoint_target,
                    games_per_opponent=games_per_opponent,
                    on_log=on_log,
                    on_progress=on_progress,
                    on_result=on_result,
                    stop_event=self._stop_event,
                )
                if self._stop_event.is_set():
                    self.event_queue.put({"type": "bench_status", "status": "stopped"})
                else:
                    self.event_queue.put({"type": "bench_status", "status": "completed"})
            except Exception as e:
                err = f"Benchmark Error: {e}\n{traceback.format_exc()}"
                self.event_queue.put({"type": "log", "target": "bench", "text": err})
                self.event_queue.put({"type": "bench_status", "status": "error", "error": str(e)})
            finally:
                self.is_active = False

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return True


class BackgroundWorker:
    """Generic worker for running callable in background thread."""
    def __init__(self, kind: str, run_fn: Callable[..., Any], event_queue: queue.Queue):
        self.kind = kind
        self.run_fn = run_fn
        self.event_queue = event_queue
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.pace_delay_sec: float = 0.0

    def set_pace_delay(self, delay: float) -> None:
        self.pace_delay_sec = max(0.0, float(delay))

    def _emit(self, type_: str, data: Any = None) -> None:
        self.event_queue.put({"kind": self.kind, "type": type_, "data": data})

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def request_stop(self) -> None:
        if self.is_running():
            self._stop_event.set()
            self._emit("log", "Stop requested... waiting for current step/game to finish.")

    def start(self, **kwargs: Any) -> bool:
        if self.is_running():
            return False

        self._stop_event.clear()

        def on_event(data: Any) -> None:
            self._emit("train_viz", data)
            if self.pace_delay_sec > 0 and isinstance(data, dict) and data.get("type") == "move":
                time.sleep(self.pace_delay_sec)

        callbacks = {
            "on_log": lambda msg: self._emit("log", msg),
            "on_progress": lambda data: self._emit("progress", data),
            "on_result": lambda data: self._emit("result", data),
            "on_event": on_event,
            "stop_event": self._stop_event,
        }
        accepted = inspect.signature(self.run_fn).parameters
        call_kwargs = {**kwargs, **{k: v for k, v in callbacks.items() if k in accepted}}

        def run() -> None:
            self._emit("status", "running")
            try:
                self.run_fn(**call_kwargs)
                self._emit("status", "stopped" if self._stop_event.is_set() else "completed")
            except Exception as e:
                self._emit("log", f"Error: {e}\n{traceback.format_exc()}")
                self._emit("status", "error")

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return True
