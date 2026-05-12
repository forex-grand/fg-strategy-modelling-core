from src.schemas import TARGET_MODEL_TYPES
from src.pipeline.trainer import Trainer as TR
from src.schemas import SymbolIn
from src.pipeline.preprocessing.base_preprocessor import PreprocessBase

def train_publish_models(preprocess_class:PreprocessBase, target_model:TARGET_MODEL_TYPES, sequence_length:int, run_performance_test: bool=True):
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

  train_model_types = ['lstm','cnn-bi-lstm','simple-ns','conservative-ns','complex-ns']

  trainer = TR(symbols=symbols,sequence_length=sequence_length, model_types=train_model_types,
            preprocessor_class=preprocess_class, target_model_type=target_model, run_performance_test=run_performance_test)
  results = trainer.run()