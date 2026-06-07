from openfe import OpenFE, transform
from sklearn.impute import SimpleImputer
import os
import joblib
import numpy as np
import pandas as pd

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
        transformer = ofe.fit(data=X_train, label=y_train, n_jobs=cpu_counts, verbose=False)
        joblib.dump(transformer, transformer_path)

    # Transform
    cpu_counts = os.cpu_count()
    X_train_new, X_test_new = transform(X_train, X_eval, transformer, n_jobs=cpu_counts)

    # Clean infinities
    X_train_new = X_train_new.replace([np.inf, -np.inf], np.nan)
    X_test_new  = X_test_new.replace([np.inf, -np.inf], np.nan)

    # Impute
    imputer = SimpleImputer(strategy="median")
    X_train_new = pd.DataFrame(imputer.fit_transform(X_train_new), columns=X_train_new.columns)
    X_test_new  = pd.DataFrame(imputer.transform(X_test_new), columns=X_test_new.columns)

    return X_train_new, X_test_new, transformer

def transform_fe(X, feature_transformer):
  cpu_counts = os.cpu_count()    
  X_transformed, _ = transform(X, X, feature_transformer, n_jobs=cpu_counts)
  return X_transformed
