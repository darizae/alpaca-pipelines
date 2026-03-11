"""
Asynchronous file reader and writer using multiprocessing.

Ported from ANIMAL-SPOT ``utils/FileIO.py`` (Bergler & Schroeter, GPL-3.0).
Platform note: multiprocessing workers are only started on Linux; on other
platforms the reader/writer falls back to synchronous I/O.
"""

from __future__ import annotations

import os
import platform
import queue
from typing import Any, Callable

import torch.multiprocessing as mp


def _default_read_fn(file_name: str) -> bytes:
    with open(file_name, "rb") as file_handle:
        return file_handle.read()


def _default_write_fn(file_name: str, data: bytes) -> None:
    with open(file_name, "wb") as file_handle:
        file_handle.write(data)


class AsyncFileReader:
    """Read files asynchronously via a pool of multiprocessing workers.

    On non-Linux platforms, falls back to synchronous reads.
    """

    def __init__(
        self,
        n_readers: int = 1,
        read_fn: Callable[[str], bytes] = _default_read_fn,
        n_retries: int = 3,
    ) -> None:
        self._read_fn = read_fn
        self.n_retries = n_retries
        self._use_async = platform.system() == "Linux"

        if self._use_async:
            self._read_queue: mp.Queue[str | None] = mp.Queue()
            self._out_queue: mp.Queue[tuple[str, bytes | None]] = mp.Queue()
            self._manager = mp.Manager()
            self._buf: dict[str, bytes | None] = self._manager.dict()
            self._read_workers = [
                mp.Process(
                    target=self._read_worker,
                    args=(self._read_queue, self._out_queue),
                    daemon=True,
                )
                for _ in range(n_readers)
            ]
            for worker in self._read_workers:
                worker.start()

    def __call__(self, file_name: str) -> bytes | None:
        if not self._use_async:
            return self._read_fn(file_name)

        self._read_queue.put(file_name)
        n_tries = 0
        while file_name not in self._buf:
            try:
                if n_tries > self.n_retries:
                    return _default_read_fn(file_name)
                fetched_name, data = self._out_queue.get(timeout=0.5)
                if fetched_name == file_name:
                    return data
                self._buf[fetched_name] = data
            except queue.Empty:
                n_tries += 1
        return self._buf.pop(file_name)

    def _read_worker(
        self,
        in_queue: mp.Queue[str | None],
        out_queue: mp.Queue[tuple[str, bytes | None]],
    ) -> None:
        while True:
            try:
                file_name = in_queue.get()
                if file_name is None:
                    break
                out_queue.put((file_name, self._read_fn(file_name)))
            except FileNotFoundError:
                out_queue.put((file_name, None))  # type: ignore[possibly-undefined]
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception:
                import traceback
                print(traceback.format_exc())
                out_queue.put((file_name, None))  # type: ignore[possibly-undefined]


class AsyncFileWriter:
    """Write files asynchronously via a pool of multiprocessing workers.

    On non-Linux platforms, falls back to synchronous writes.
    """

    def __init__(
        self,
        write_fn: Callable[[str, Any], None] = _default_write_fn,
        n_writers: int = 1,
    ) -> None:
        self._write_fn = write_fn
        self._use_async = platform.system() == "Linux"

        if self._use_async:
            self._write_queue: mp.Queue[tuple[str, Any] | None] = mp.Queue()
            self._write_workers = [
                mp.Process(
                    target=self._write_worker,
                    args=(self._write_queue,),
                    daemon=True,
                )
                for _ in range(n_writers)
            ]
            for worker in self._write_workers:
                worker.start()

    def __call__(self, file_name: str, data: Any) -> None:
        if not self._use_async:
            self._write_fn(file_name, data)
            return
        self._write_queue.put((file_name, data))

    def _write_worker(self, in_queue: mp.Queue[tuple[str, Any] | None]) -> None:
        while True:
            try:
                item = in_queue.get()
                if item is None:
                    break
                file_name, data = item
                self._write_fn(file_name, data)
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception:
                import traceback
                print(traceback.format_exc())
