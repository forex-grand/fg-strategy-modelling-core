from src.schemas import TARGET_MODEL_TYPES
from src.pipeline.trainer import Trainer as TR
from src.schemas import SymbolIn
from src.pipeline.preprocessing.base_preprocessor import PreprocessBase


def train_publish_xgb_models(preprocess_class:PreprocessBase, target_model:TARGET_MODEL_TYPES, 
                  sequence_length:int, run_performance_test: bool=True, hot_reload: bool=True):
  symbols = [
    SymbolIn(symbol="AUDUSD", group="forex"),
    SymbolIn(symbol="EURUSD", group="forex"),
    SymbolIn(symbol="GBPUSD", group="forex"),
    SymbolIn(symbol="NZDUSD", group="forex"),
    SymbolIn(symbol="USDCAD", group="forex"),
    SymbolIn(symbol="USDCHF", group="forex"),
    SymbolIn(symbol="USDJPY", group="forex"),
    SymbolIn(symbol="XAUUSD", group="metals"),
    ]

  train_model_types = [
      # ── XGBoost models ───────────────────────────────
      'xgb-simple', 'xgb-tiny', 'xgb-simple-shallow', 'xgb-simple-slow',
      'xgb-balanced', 'xgb-l1-regularised', 'xgb-l2-regularised', 'xgb-gamma-pruned',
      'xgb-column-sampled', 'xgb-deep-trees', 'xgb-high-capacity',
      'xgb-elastic-net', 'xgb-high-child-weight', 'xgb-max-complex',
  ]
  trainer = TR(symbols=symbols,sequence_length=sequence_length, model_types=train_model_types,
            preprocessor_class=preprocess_class, target_model_type=target_model, 
            run_performance_test=run_performance_test, hot_reload_data=hot_reload,
            )
  results = trainer.run()
