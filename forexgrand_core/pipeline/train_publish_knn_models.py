import os

from forexgrand_core.pipeline.knn_trainer import KNNTrainer
from forexgrand_core.pipeline.preprocessing.base_preprocessor import PreprocessBase
from forexgrand_core.schemas import SymbolIn


def train_publish_knn_models(
    preprocess_class: PreprocessBase,
    sequence_length: int,
    upload_models: bool = False,
    hot_reload: bool = True,
    use_dataframe_format: bool = False,
):
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

    symbols = []
    source = os.getenv("DATA_SOURCE")
    if source == "metaquotes":
        symbols = metaquoutes
    elif source == "deriv":
        symbols = deriv

    trainer = KNNTrainer(
        symbols=symbols,
        sequence_length=sequence_length,
        preprocessor_class=preprocess_class,
        hot_reload_data=hot_reload,
        upload_models=upload_models,
        use_dataframe_format=use_dataframe_format,
    )
    return trainer.run()
