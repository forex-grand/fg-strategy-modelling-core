"""Command line interface for the ForexGrand modelling core."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def _load_input(path: Path) -> Any:
    suffixes = path.suffixes
    if ".parquet" in suffixes:
        return pd.read_parquet(path)
    if ".csv" in suffixes:
        return pd.read_csv(path)
    if path.suffix in {".pkl", ".pickle"} or ".pkl.gz" in suffixes or ".pickle.gz" in suffixes:
        return pd.read_pickle(path, compression="infer")
    raise ValueError("Input must be a CSV, Parquet, or pickle file (optionally gzip-compressed).")


def _load_preprocess_fn(path: Path, function_name: str) -> Callable[[Any], Any]:
    module_name = f"fg_core_preprocessor_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load preprocessing module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    preprocess_fn = getattr(module, function_name, None)
    if not callable(preprocess_fn):
        raise ValueError(f"{path} must define a callable '{function_name}'.")
    return preprocess_fn


def _save_gzip_pickle(value: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wb") as file_handle:
        pickle.dump(value, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def _default_preprocessed_path(input_path: Path, output: Path | None) -> Path:
    if output is None:
        return input_path.with_name(f"{input_path.stem}.preprocessed.pkl.gz")
    if output.exists() and output.is_dir():
        return output / f"{input_path.stem}.preprocessed.pkl.gz"
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fg_core", description="ForexGrand modelling data tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download_data", help="Download and cache symbol market data")
    download.add_argument("symbol_pair")
    download.add_argument("instrument_group")
    download.add_argument("--bucket")
    download.add_argument("--source")
    download.set_defaults(handler=_download_data)

    generate = subparsers.add_parser("generate_train_data", help="Generate train and eval data")
    generate.add_argument("symbol_pair")
    generate.add_argument("instrument_group")
    generate.add_argument("--sequence-length", type=int, required=True)
    generate.add_argument("--stride", type=int, required=True)
    generate.add_argument("--train-bucket")
    generate.add_argument("--eval-bucket")
    generate.add_argument("--source")
    generate.add_argument("--hot-reload", action="store_true")
    generate.add_argument("--dataframe-format", action="store_true")
    generate.add_argument("--target-stop-minutes", type=int)
    generate.add_argument("--target-mode", choices=("points", "raw_difference", "prices"), default="points")
    generate.set_defaults(handler=_generate_train_data)

    preprocess = subparsers.add_parser("preprocess_data", help="Apply a Python preprocess_fn and save gzip output")
    preprocess.add_argument("input", type=_path)
    preprocess.add_argument("preprocess_file", type=_path)
    preprocess.add_argument("--output", type=Path)
    preprocess.add_argument("--function", default="preprocess_fn")
    preprocess.set_defaults(handler=_preprocess_data)

    backtest = subparsers.add_parser("run_backtest", help="Run a strategy against market data")
    backtest.add_argument("strategy_path", type=_path)
    backtest.add_argument("symbol_pair")
    backtest.add_argument("--instrument-group", help="DataManager instrument group")
    backtest.add_argument("--bucket", required=True, help="DataManager storage bucket")
    backtest.add_argument("--source", required=True, help="Market-data source")
    backtest.add_argument("--sequence-length", type=int, default=60, help="Strategy window length (default: 60)")
    backtest.add_argument("--stride", type=int, default=1, help="Bars between strategy windows (default: 1)")
    backtest.add_argument("--batch-size", type=int, default=1024, help="Strategy windows per batch (default: 1024)")
    backtest.add_argument(
        "--sl-calculation",
        type=json.loads,
        metavar="JSON",
        help=("SL/TP JSON: fixed={mode,sl_points,tp_points}; "
              "range={mode,range,sl_ratio,tp_ratio}; "
              "atr={mode,sl_multiplier,tp_multiplier,atr_period}"),
    )
    backtest.add_argument("--entry-price-type", choices=("bid", "ask", "mid"), default="bid", help="Entry price convention (default: bid)")
    backtest.add_argument("--start-index", type=int, default=0, help="Inclusive first market row (default: 0)")
    backtest.add_argument("--end-index", type=int, default=-1, help="Exclusive last market row; -1 means final row")
    backtest.add_argument("--return-in-points", action="store_true", help="Return profits and drawdowns in symbol points")
    backtest.add_argument("--output", type=Path, help="Save the complete result as a gzip pickle")
    backtest.set_defaults(handler=_run_backtest)
    return parser


def _download_data(args: argparse.Namespace) -> dict[str, Any]:
    from forexgrand_core.data_manager import DataManager

    manager = DataManager(base_bucket_name=args.bucket, source=args.source)
    dataframe, properties = manager.load_data(args.symbol_pair, args.instrument_group)
    parquet_path = manager._build_local_directory(
        args.instrument_group.strip(), args.symbol_pair.strip()
    ) / f"{args.symbol_pair.strip()}_M1.parquet"
    return {
        "command": "download_data",
        "data_file": str(parquet_path),
        "rows": len(dataframe),
        "symbol": properties.symbol,
    }


def _generate_train_data(args: argparse.Namespace) -> dict[str, Any]:
    from forexgrand_core.generate_train_data import GenerateTrainData
    from forexgrand_core.schemas import TimeBasedTarget

    target_model = None
    if args.target_stop_minutes is not None:
        target_model = TimeBasedTarget(stop_minutes=args.target_stop_minutes, mode=args.target_mode)
    generator = GenerateTrainData(
        train_base_bucket=args.train_bucket or "forexgrand-train",
        eval_base_bucket=args.eval_bucket or "forexgrand-eval",
        use_dataframe_format=args.dataframe_format,
    )
    train_path, eval_path = generator.load_data(
        args.symbol_pair,
        args.instrument_group,
        sequence_length=args.sequence_length,
        stride=args.stride,
        hot_reload=args.hot_reload,
        target_model=target_model,
        source=args.source,
    )
    return {"command": "generate_train_data", "train_output": str(train_path), "eval_output": str(eval_path)}


def _preprocess_data(args: argparse.Namespace) -> dict[str, Any]:
    preprocess_fn = _load_preprocess_fn(args.preprocess_file, args.function)
    input_data = _load_input(args.input)
    result = preprocess_fn(input_data)
    output_path = _default_preprocessed_path(args.input, args.output)
    _save_gzip_pickle(result, output_path)
    return {"command": "preprocess_data", "output": str(output_path), "type": type(result).__name__}


def _run_backtest(args: argparse.Namespace) -> dict[str, Any]:
    from forexgrand_core.backtesting import run_backtest

    result = run_backtest(
        args.strategy_path,
        bucket_name=args.bucket,
        source=args.source,
        symbol_pair=args.symbol_pair,
        instrument_group=args.instrument_group,
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        sl_calculation=args.sl_calculation,
        entry_price_type=args.entry_price_type,
        start_index=args.start_index,
        end_index=args.end_index,
        return_in_points=args.return_in_points,
    )
    output = None
    if args.output is not None:
        output = str(_save_gzip_pickle(result, args.output))
    return {
        "command": "run_backtest",
        "output": output,
        "positions_total": result.positions_total,
        "buy_count": result.buy_count,
        "sell_count": result.sell_count,
        "unsupported_signal_count": result.unsupported_signal_count,
        "profit_equity_points": result.profit_equity.tolist() if args.return_in_points else None,
        "dd_equity_points": result.dd_equity.tolist() if args.return_in_points else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (OSError, ValueError, ImportError) as error:
        raise SystemExit(f"fg_core: error: {error}") from error
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    main()