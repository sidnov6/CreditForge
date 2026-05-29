"""Calibration — turn rankings into true probabilities (Part 3.2).

A model can rank perfectly yet output miscalibrated probabilities, which makes
the Expected-Loss/capital numbers wrong. We fit the calibrator on a held-out
**calibration** slice (never train, never test) so the mapping is honest.

Model-agnostic by design: it maps any model's raw PD -> calibrated PD, so the
scorecard and the challenger are calibrated through the identical mechanism.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from creditforge.config import Config, load_config, run_stamp


class Calibrator:
    """Maps raw predicted PD -> calibrated PD via isotonic or Platt scaling."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._iso: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None

    def fit(self, raw_p: np.ndarray, y: np.ndarray) -> "Calibrator":
        raw_p = np.asarray(raw_p, float)
        y = np.asarray(y, int)
        if self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            self._iso.fit(raw_p, y)
        elif self.method in ("sigmoid", "platt"):
            # Platt scaling: logistic regression on the logit of the raw score
            z = np.log(np.clip(raw_p, 1e-6, 1 - 1e-6) / (1 - np.clip(raw_p, 1e-6, 1 - 1e-6)))
            self._platt = LogisticRegression().fit(z.reshape(-1, 1), y)
        else:
            raise ValueError(f"unknown calibration method: {self.method}")
        return self

    def transform(self, raw_p: np.ndarray) -> np.ndarray:
        raw_p = np.asarray(raw_p, float)
        if self._iso is not None:
            return self._iso.transform(raw_p)
        if self._platt is not None:
            z = np.log(np.clip(raw_p, 1e-6, 1 - 1e-6) / (1 - np.clip(raw_p, 1e-6, 1 - 1e-6)))
            return self._platt.predict_proba(z.reshape(-1, 1))[:, 1]
        raise RuntimeError("Calibrator not fitted")


def fit_calibrator(raw_p: np.ndarray, y: np.ndarray,
                   cfg: Config | None = None) -> Calibrator:
    cfg = cfg or load_config()
    return Calibrator(method=str(cfg.calibration.method)).fit(raw_p, y)


def save(calibrator: Calibrator, name: str, cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    out = cfg.path("artifacts") / f"calibrator_{name}.joblib"
    joblib.dump({"calibrator": calibrator, "stamp": run_stamp()}, out)
    return out


def load(name: str, cfg: Config | None = None) -> Calibrator:
    cfg = cfg or load_config()
    return joblib.load(cfg.path("artifacts") / f"calibrator_{name}.joblib")["calibrator"]
