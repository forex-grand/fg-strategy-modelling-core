from openfe import OpenFE, transform
from sklearn.impute import SimpleImputer
import os
import joblib
import numpy as np
import pandas as pd
import warnings


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
        verbose = int(os.getenv('OPENFE_VERBOSE',"500"))
        transformer = ofe.fit(data=X_train, label=y_train, n_jobs=cpu_counts, verbose=verbose)

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
