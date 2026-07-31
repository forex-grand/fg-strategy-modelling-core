import os

from forexgrand_core.pipeline.no_train_trainer import NoTrainTrainer
from forexgrand_core.pipeline.preprocessing.base_preprocessor import PreprocessBase
from forexgrand_core.schemas import TARGET_MODEL_TYPES, SymbolIn


def publish_no_train_model(preprocess_class: PreprocessBase, target_model: TARGET_MODEL_TYPES,
                  sequence_length: int, run_performance_test: bool = True, upload_models: bool = False,
                  hot_reload: bool = True, target_percentile: float = 75, use_dataframe_format: bool = False):
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

  trainer = NoTrainTrainer(
      symbols=symbols,
      sequence_length=sequence_length,
      preprocessor_class=preprocess_class,
      target_model_type=target_model,
      run_performance_test=run_performance_test,
      hot_reload_data=hot_reload,
      upload_models=upload_models,
      target_percentile=target_percentile,
      use_dataframe_format=use_dataframe_format,
  )
  return trainer.run()
