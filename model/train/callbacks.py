"""Training callbacks (early stopping, etc.)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Mapping, Optional

from model.train.config import CellVOCALTrainConfig

logger = logging.getLogger(__name__)

MonitorMode = Literal["min", "max"]


@dataclass(frozen=True)
class EarlyStoppingState:
    """Snapshot of early-stopping internal state (for logging / tests)."""

    best_score: Optional[float]
    best_epoch: int
    wait: int
    stopped_epoch: Optional[int]


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    API mirrors common DL frameworks (Keras / PyTorch Lightning): call ``step``
    after each epoch; it returns ``True`` when training should halt.
    """

    def __init__(
        self,
        *,
        monitor: str = "loss",
        patience: int = 20,
        min_delta: float = 0.0,
        mode: MonitorMode = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode: MonitorMode = mode

        self._enabled = patience > 0
        self._best_score: Optional[float] = None
        self._best_epoch: int = -1
        self._wait: int = 0
        self._stopped_epoch: Optional[int] = None
        self._warned_missing_monitor = False

    @classmethod
    def from_train_config(cls, cfg: CellVOCALTrainConfig) -> EarlyStopping:
        return cls(
            monitor=cfg.early_stop_metric,
            patience=cfg.early_stop_patience,
            min_delta=cfg.early_stop_min_delta,
            mode=cfg.early_stop_mode,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def state(self) -> EarlyStoppingState:
        return EarlyStoppingState(
            best_score=self._best_score,
            best_epoch=self._best_epoch,
            wait=self._wait,
            stopped_epoch=self._stopped_epoch,
        )

    def step(self, metrics: Mapping[str, float], epoch: int) -> bool:
        """
        Update state from epoch metrics.

        Returns
        -------
        bool
            ``True`` if training should stop.
        """
        if not self._enabled:
            return False

        if self.monitor not in metrics:
            if not self._warned_missing_monitor:
                logger.warning(
                    "EarlyStopping: monitor %r not in metrics keys %s; skipping checks",
                    self.monitor,
                    sorted(metrics.keys()),
                )
                self._warned_missing_monitor = True
            return False

        current = float(metrics[self.monitor])
        if self._is_improvement(current):
            self._best_score = current
            self._best_epoch = epoch
            self._wait = 0
            return False

        self._wait += 1
        if self._wait >= self.patience:
            self._stopped_epoch = epoch
            logger.info(
                "Early stopping at epoch %d: %r did not improve in %d epochs "
                "(best=%.6g @ epoch %d)",
                epoch,
                self.monitor,
                self.patience,
                self._best_score,
                self._best_epoch,
            )
            return True
        return False

    def _is_improvement(self, current: float) -> bool:
        if self._best_score is None:
            return True
        if self.mode == "min":
            return current < self._best_score - self.min_delta
        return current > self._best_score + self.min_delta
