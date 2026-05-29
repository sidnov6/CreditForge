"""MLflow experiment tracking + registry (Part 10.2).

Self-hosted, free, local file store. Logs the run stamp, key config params, the
out-of-time metrics, and the artifacts directory so any model version is
re-instantiable. Tracking is best-effort: if MLflow misbehaves it must never
break the training pipeline.
"""
from __future__ import annotations

from contextlib import contextmanager

from creditforge.config import REPO_ROOT, Config, load_config, run_stamp


@contextmanager
def log_run(run_name: str, cfg: Config | None = None):
    cfg = cfg or load_config()
    try:
        import mlflow
    except ImportError:
        yield None
        return

    mlflow.set_tracking_uri(f"file:{REPO_ROOT / 'mlruns'}")
    mlflow.set_experiment("creditforge")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(run_stamp())
        yield _Logger(mlflow, run)


class _Logger:
    def __init__(self, mlflow, run):
        self._mlflow = mlflow
        self.run = run

    def params(self, **kwargs):
        try:
            self._mlflow.log_params({k: str(v) for k, v in kwargs.items()})
        except Exception:
            pass

    def metrics(self, **kwargs):
        try:
            self._mlflow.log_metrics({k: float(v) for k, v in kwargs.items()
                                      if v is not None})
        except Exception:
            pass

    def artifacts(self, path):
        try:
            self._mlflow.log_artifacts(str(path))
        except Exception:
            pass
