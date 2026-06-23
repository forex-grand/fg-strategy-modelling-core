from openfe import OpenFE, transform
from sklearn.impute import SimpleImputer
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import os
import sys
import joblib
import numpy as np
import pandas as pd
import warnings


@contextmanager
def _suppress_native_output():
    """Silence C/C++ extension output, including logs emitted by child processes."""
    stdout_fd = 1
    stderr_fd = 2
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        sys.stdout.flush()
        sys.stderr.flush()
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _openfe_stage2_params(n_jobs):
    return {
        "n_estimators": 1000,
        "importance_type": "gain",
        "num_leaves": 16,
        "seed": 1,
        "n_jobs": n_jobs,
        "verbosity": -1,
        "force_col_wise": True,
    }


def _is_transformer_bundle(feature_transformer):
    return isinstance(feature_transformer, dict) and "features" in feature_transformer


def _get_openfe_features(feature_transformer):
    if _is_transformer_bundle(feature_transformer):
        return feature_transformer["features"]
    return feature_transformer


def _get_openfe_imputer(feature_transformer):
    if _is_transformer_bundle(feature_transformer):
        return feature_transformer.get("imputer")
    return None


def _openfe_transform(X_train, X_eval, feature_transformer, n_jobs):
    features = _get_openfe_features(feature_transformer)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="overflow encountered in exp",
            category=RuntimeWarning,
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return transform(X_train, X_eval, features, n_jobs=n_jobs)


def _sanitize_openfe_frame(frame):
    return frame.replace([np.inf, -np.inf], np.nan)


def _make_transformer_bundle(features, imputer):
    return {"features": features, "imputer": imputer}

def auto_expand_feature_fe(X_train, y_train, X_eval, metadata={}, force_reload=False):
    # Build save path from metadata
    meta_str = "_".join(f"{k}-{v}" for k, v in sorted(metadata.items()))
    save_dir = os.path.join("data", meta_str if meta_str else "default")
    os.makedirs(save_dir, exist_ok=True)
    transformer_path = os.path.join(save_dir, "transformer.pkl")

    
    if os.path.exists(transformer_path) and not force_reload:
        print(f"Loading transformer from {transformer_path}")
        transformer = joblib.load(transformer_path)
    else:
        print(f"Fitting transformer, will save to {transformer_path}")
        cpu_counts = os.cpu_count()
        ofe = OpenFE()
        verbose = _env_flag("OPENFE_VERBOSE", default=False)
        output_context = nullcontext() if verbose else _suppress_native_output()
        with output_context:
            transformer = ofe.fit(
                data=X_train,
                label=y_train,
                n_jobs=cpu_counts,
                verbose=verbose,
                stage2_params=_openfe_stage2_params(cpu_counts),
            )

    # Transform
    cpu_counts = os.cpu_count()
    X_train_new, X_test_new = _openfe_transform(X_train, X_eval, transformer, n_jobs=cpu_counts)

    # Clean infinities
    X_train_new = _sanitize_openfe_frame(X_train_new)
    X_test_new = _sanitize_openfe_frame(X_test_new)

    # Impute
    imputer = _get_openfe_imputer(transformer)
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        X_train_new = pd.DataFrame(imputer.fit_transform(X_train_new), columns=X_train_new.columns)
        transformer = _make_transformer_bundle(_get_openfe_features(transformer), imputer)
        joblib.dump(transformer, transformer_path)
    else:
        X_train_new = pd.DataFrame(imputer.transform(X_train_new), columns=X_train_new.columns)
    X_test_new = pd.DataFrame(imputer.transform(X_test_new), columns=X_test_new.columns)

    return X_train_new, X_test_new, transformer

def transform_fe(X, feature_transformer):
  cpu_counts = os.cpu_count()
  X_transformed, _ = _openfe_transform(X, X, feature_transformer, n_jobs=cpu_counts)
  X_transformed = _sanitize_openfe_frame(X_transformed)
  imputer = _get_openfe_imputer(feature_transformer)
  if imputer is not None:
    X_transformed = pd.DataFrame(imputer.transform(X_transformed), columns=X_transformed.columns)
  return X_transformed
