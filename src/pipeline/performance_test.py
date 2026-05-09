import tensorflow as tf
import pandas as pd
import tempfile
import asyncio
import requests
import json
from src.models_architecture.base_model import BaseModel
from src.settings import Settings
from src.pipeline.generate_train_data import GenerateTrainData

trade_managers = [
    ###fixed stop loss and take profit points
    # {"type":"fixed_sl_tp","params":{"stop_loss_points":50,"take_profit_points":100}},
    # {"type":"fixed_sl_tp","params":{"stop_loss_points":100,"take_profit_points":200}},
    # {"type":"fixed_sl_tp","params":{"stop_loss_points":100,"take_profit_points":100}},
    {"type":"fixed_sl_tp","params":{"stop_loss_points":200,"take_profit_points":200}},
    {"type":"fixed_sl_tp","params":{"stop_loss_points":200,"take_profit_points":100}},

    # # ##RISK REWARD BASED
    # {"type":"risk_reward", "params":{"rr_ratio":3.0,
    #                         "sl_calculator":{"type":"fixed","sl_points":200}}},
    # {"type":"risk_reward", "params":{"rr_ratio":3.0,
    #             "sl_calculator":{"type":"atr","atr_period":14,"atr_multiplier":1.5,"atr_timeframe":"M5"}}},
    # {"type":"risk_reward", "params":{"rr_ratio":3.0,
    #             "sl_calculator":{"type":"stdev", "stdev_period":20,"stdev_multiplier":1.0,"stdev_timeframe":"M5"}}},
    # {"type":"risk_reward", "params":{"rr_ratio":3.0,
    #             "sl_calculator":{"type":"swing","swing_bars":14,"swing_timeframe":"M15"}}},
    ###SL DISTANCE TRIGGER
    {"type":"chandelier_exit", "params":{"period":14,"multiplier":3.0,"timeframe":"M5","use_close":False}},
    {"type":"chandelier_exit", "params":{"period":7,"multiplier":2.0,"timeframe":"M5","use_close":False}},

    # ###BREAKEVEN STOP
    # {"type":"breakeven_stop", "params":{"lock_in_points":10,"trigger":{
    #     "type":"atr_trigger", "atr_period":14, "atr_multiplier":2.0, "timeframe": "M5"}}},
    # {"type":"breakeven_stop", "params":{"lock_in_points":10,"trigger":{
    #     "type":"sl_distance_trigger", "sl_ratio":2}}},

    # ##MULTIPLE TARGETS
    # {"type":"multiple_targets", "params":{"tp1_fraction":0.33,"tp2_fraction":0.33, "target_config":{"type":"atr_distance", "timeframe":"M5"}}},
    # {"type":"multiple_targets", "params":{"tp1_fraction":0.33,"tp2_fraction":0.33, "target_config":{"type":"sl_ratio_distance"}}},

    # ##TIME STOP BASED
    {"type":"time_stop", "params": {"max_duration_minutes":60}},
    {"type":"time_stop", "params": {"max_duration_minutes":240}},
    {"type":"time_stop", "params": {"max_duration_minutes":360}},
    {"type":"time_stop", "params": {"max_duration_minutes":720}},
    ]

lotsizers = [
    ##fixed lotsize
    # {"type":"fixed_lot_size", "params":{"fixed_lots":1.0}},

    ##percentage risk
    {"type":"percentage_risk", "params":{"risk_pct":0.02, "min_lots":0.01, "max_lots":1}},

    ##VOLATILITY LOTSIZER
    {"type":"volatility_lot_size", "params":{"atr_period":14, "atr_timeframe":"M5", "risk_pct":0.02, "atr_multiplier":1.0, "min_risk_pct":0.01, "max_risk_pct":0.5, "abs_min_lots":0.01}},

    # ##MARTINGALE LOTSIZER
    {"type":"martingale_lot_size", "params":{"base_lots":0.01, "multiplier":1.5, "max_steps":5}},

    {"type":"anti_martingale_lot_size", "params":{"base_lots":0.01, "multiplier":1.5, "max_steps":5}}

]


def test_model_live_performance(
        model: BaseModel,
        symbol: str,
        group: str,
        sequence_length: int,
        config: Settings,
        model_id: str,
        stride: int = 1
    ):
    data_bucket = config.test_bucket_name.strip()
    data_gen = GenerateTrainData(eval_base_bucket=data_bucket,
                                 train_base_bucket=data_bucket)
    seq_data_path = data_gen.load_single_data(
        bucket_name=data_bucket,
        symbol_pair=symbol,
        instrument_group=group,
        sequence_length=sequence_length,
        stride=stride,
        hot_reload=False
    )

    load_data = tf.data.TFRecordDataset(seq_data_path, compression_type=config.tf_record_compression_type)
    model = model.get_serving_signature()
    batch_dataset = load_data.batch(128).take(-1)

    df_dict = {
        'timestamp': [],
        'signal_type': [],
        'price': [],
    }

    for batch in batch_dataset:
        predictions = model(batch)['output']
        batch_data = tf.io.parse_example(batch, features={
            'time': tf.io.FixedLenFeature([sequence_length], tf.int64),
            'close': tf.io.FixedLenFeature([sequence_length], tf.float32)
        })
        batch_times = batch_data['time'][:, -1]
        batch_prices = batch_data['close'][:, -1]
        pred_types = tf.where(predictions == 0, "BUY", tf.where(predictions == 1, "SELL", "HOLD"))
        valid_signals = tf.reshape(tf.where(tf.not_equal(predictions, 2)), [-1])
        df_dict['signal_type'].extend([t.decode('utf-8') for t in tf.gather(pred_types, valid_signals).numpy()])
        df_dict['timestamp'].extend(tf.gather(batch_times, valid_signals).numpy())
        df_dict['price'].extend(tf.gather(batch_prices, valid_signals).numpy())

    df = pd.DataFrame(df_dict)
    print("total signals: ", len(df))

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as temp_file:
        df.to_csv(temp_file.name, index=False)
        temp_path = temp_file.name

    with open(temp_path, 'rb') as f:
        performance_test_config = {
            "strategy": {
                "source": "csv",
                "type": "",
                "params": {},
            },
            "symbols": [symbol],
            "trade_managers": trade_managers,
            "lotsizers": lotsizers,
            "data_source": "metaquotes",
            "data_bucket": data_bucket,
            "group":group,
            "name": f"{model_id}_{symbol}"
        }

        response = requests.post(
            f'{config.performance_base_url}/performance-tests/uploaded-signals',
            data={'config': json.dumps(performance_test_config)},
            files={'file': (f'{model_id}_{symbol}_signals.csv', f, 'text/csv')}
        )

        resp_json = response.json()
        print("Test Job Id: ", resp_json.get('job_id', resp_json), " Status: ", response.status_code)

    return df