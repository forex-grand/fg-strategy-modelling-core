import os

from src.schemas import TARGET_MODEL_TYPES
from src.pipeline.trainer import Trainer as TR
from src.schemas import SymbolIn
from src.pipeline.preprocessing.base_preprocessor import PreprocessBase


def publish_no_train_model(preprocess_class:PreprocessBase, target_model:TARGET_MODEL_TYPES, 
                  sequence_length:int, run_performance_test: bool=True, upload_models: bool = False,
                  hot_reload: bool=True, target_percentile: float=75, use_dataframe_format=False):
  symbols = []
  metaquoutes = [
    SymbolIn(symbol="AUDUSD", group="forex"),
    SymbolIn(symbol="EURUSD", group="forex"),
    SymbolIn(symbol="GBPUSD", group="forex"),
    SymbolIn(symbol="NZDUSD", group="forex"),
    SymbolIn(symbol="USDCAD", group="forex"),
    SymbolIn(symbol="USDCHF", group="forex"),
    SymbolIn(symbol="USDJPY", group="forex"),
    SymbolIn(symbol="XAUUSD", group="metals"),
    ]
  
  deriv = [
    SymbolIn(symbol="Volatility 10 Index", group="volatility_indices"),
    SymbolIn(symbol="Volatility 25 Index", group="volatility_indices"),
    SymbolIn(symbol="Volatility 50 Index", group="volatility_indices"),
    SymbolIn(symbol="Volatility 75 Index", group="volatility_indices"),
    SymbolIn(symbol="Volatility 100 Index", group="volatility_indices"),
  ]

  source = os.getenv("DATA_SOURCE")
  if source == "metaquotes":
    symbols = metaquoutes
  elif source == "deriv":
    symbols = deriv

  train_model_types = ['no-train']

  trainer = TR(symbols=symbols,sequence_length=sequence_length, model_types=train_model_types,
            preprocessor_class=preprocess_class, target_model_type=target_model, 
            run_performance_test=run_performance_test, hot_reload_data=hot_reload,
            upload_models=upload_models,target_percentile=target_percentile, 
            use_dataframe_format=use_dataframe_format
            )
  results = trainer.run()
