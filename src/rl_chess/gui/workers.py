"""Background worker threads for training and benchmarking with UI queue messaging."""

from __future__ import annotations

import queue
import threading
import traceback
from typing import Any, Dict, Optional

from rl_chess.eval.benchmark import run_benchmark
from rl_chess.train.loop import run_training


class TrainingWorker:
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
            self.event_queue.put({"type": "log", "text": "Stop requested... waiting for current step/game to finish."})

    def start(self, config_path: str = "configs/tiny.yaml", generations: int = 1) -> bool:
        if self.is_running():
            return False

        self._stop_event.clear()
        self.is_active = True

        def run():
            self.event_queue.put({"type": "train_status", "status": "running"})
            try:
                def on_log(msg: str):
                    self.event_queue.put({"type": "log", "target": "train", "text": msg})

                def on_progress(data: Dict[str, Any]):
                    self.event_queue.put({"type": "train_progress", "data": data})

                run_training(
                    config_path=config_path,
                    generations=generations,
                    on_log=on_log,
                    on_progress=on_progress,
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
        config_path: str = "configs/tiny.yaml",
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
