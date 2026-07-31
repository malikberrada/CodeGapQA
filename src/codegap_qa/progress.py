from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys
from typing import Iterable, Iterator, TypeVar

from tqdm.auto import tqdm


T = TypeVar("T")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass
class ProgressManager:
    """Centralized tqdm configuration for deterministic nested progress bars."""

    enabled: bool = True
    mininterval: float = 0.2
    leave_nested: bool = False
    dynamic_ncols: bool = True

    @classmethod
    def from_config(cls, config: dict | None = None) -> "ProgressManager":
        settings = (config or {}).get("progress", {})
        enabled = bool(settings.get("enabled", True))
        enabled = _env_flag("CODEGAP_PROGRESS", enabled)
        return cls(
            enabled=enabled,
            mininterval=float(settings.get("mininterval", 0.2)),
            leave_nested=bool(settings.get("leave_nested", False)),
            dynamic_ncols=bool(settings.get("dynamic_ncols", True)),
        )

    def bar(
        self,
        iterable: Iterable[T] | None = None,
        *,
        total: int | None = None,
        desc: str,
        unit: str = "it",
        leave: bool = True,
        position: int | None = None,
        **kwargs,
    ):
        return tqdm(
            iterable=iterable,
            total=total,
            desc=desc,
            unit=unit,
            disable=not self.enabled,
            mininterval=self.mininterval,
            dynamic_ncols=self.dynamic_ncols,
            leave=leave,
            position=position,
            file=sys.stderr,
            **kwargs,
        )

    @contextmanager
    def stage(self, desc: str, *, leave: bool = True) -> Iterator[None]:
        bar = self.bar(total=1, desc=desc, unit="stage", leave=leave)
        try:
            yield
        finally:
            bar.update(1)
            bar.close()

    def write(self, message: str) -> None:
        if self.enabled:
            tqdm.write(message, file=sys.stderr)
        else:
            print(message, file=sys.stderr)


def default_progress(config: dict | None = None) -> ProgressManager:
    return ProgressManager.from_config(config)
