"""Model quality evaluator for deployment eligibility."""

from __future__ import annotations

import logging

from src.settings import Settings

LOGGER = logging.getLogger(__name__)


class Evaluator:
    """Apply strict pass/fail gates to trained model metrics."""

    def __init__(self, config: Settings) -> None:
        self.config: Settings = config

    def evaluate(self, metrics: dict[str, float]) -> tuple[bool, dict[str, str]]:
        """Return validity flag and per-check reason strings."""
        reasons: dict[str, str] = {}

        # auc = float(metrics.get("auc", 0.0))
        precision_buy = float(metrics.get("precision_buy", 0.0))
        precision_sell = float(metrics.get("precision_sell", 0.0))
        recall_buy = float(metrics.get("recall_buy", 0.0))
        recall_sell = float(metrics.get("recall_sell", 0.0))
        val_loss = float(metrics.get("val_loss", 1e9))
        train_loss = float(metrics.get("train_loss", 1e9))
        loss_gap_ratio = abs(val_loss - train_loss) / max(abs(train_loss), 1e-8)

        # reasons["auc"] = (
        #     f"PASS: {auc:.4f} > {self.config.eval_min_auc:.4f}"
        #     if auc > self.config.eval_min_auc
        #     else f"FAIL: {auc:.4f} <= {self.config.eval_min_auc:.4f}"
        # )
        reasons["precision_buy"] = (
            f"PASS: {precision_buy:.4f} > {self.config.eval_min_precision:.4f}"
            if precision_buy > self.config.eval_min_precision
            else f"FAIL: {precision_buy:.4f} <= {self.config.eval_min_precision:.4f}"
        )
        reasons["precision_sell"] = (
            f"PASS: {precision_sell:.4f} > {self.config.eval_min_precision:.4f}"
            if precision_sell > self.config.eval_min_precision
            else f"FAIL: {precision_sell:.4f} <= {self.config.eval_min_precision:.4f}"
        )
        reasons["recall_buy"] = (
            f"PASS: {recall_buy:.4f} > {self.config.eval_min_recall:.4f}"
            if recall_buy > self.config.eval_min_recall
            else f"FAIL: {recall_buy:.4f} <= {self.config.eval_min_recall:.4f}"
        )
        reasons["recall_sell"] = (
            f"PASS: {recall_sell:.4f} > {self.config.eval_min_recall:.4f}"
            if recall_sell > self.config.eval_min_recall
            else f"FAIL: {recall_sell:.4f} <= {self.config.eval_min_recall:.4f}"
        )
        reasons["overfit_gap"] = (
            f"PASS: {loss_gap_ratio:.4f} <= {self.config.eval_max_overfit_gap:.4f}"
            if loss_gap_ratio <= self.config.eval_max_overfit_gap
            else f"FAIL: {loss_gap_ratio:.4f} > {self.config.eval_max_overfit_gap:.4f}"
        )

        is_valid = all(text.startswith("PASS") for text in reasons.values())
        for key, value in reasons.items():
            LOGGER.info("Evaluator %s: %s", key, value)

        return is_valid, reasons

