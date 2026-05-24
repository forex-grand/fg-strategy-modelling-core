import tensorflow as tf
import pandas as pd
import tempfile
import asyncio
import requests
import json
from src.models_architecture.base_model import BaseModel
from src.settings import Settings
from src.pipeline.generate_train_data import GenerateTrainData
import os
import glob

point_multiplier = int(os.getenv("POINT_MULTIPLIER", 1))

trade_managers = [
    ###fixed stop loss and take profit points
    {"type":"fixed_sl_tp","params":{"stop_loss_points":200*point_multiplier,"take_profit_points":100*point_multiplier}},
    {"type":"fixed_sl_tp","params":{"stop_loss_points":200*point_multiplier,"take_profit_points":300*point_multiplier}},

    # # ##RISK REWARD BASED
    {"type":"risk_reward", "params":{"rr_ratio":3.0,
                            "sl_calculator":{"type":"fixed","sl_points":200*point_multiplier},}},
    
    {"type":"time_stop", "params": {"max_duration_minutes":240}},
    # {"type":"time_stop", "params": {"max_duration_minutes":720}},
    ]

lotsizers = [
    ##percentage risk
    {"type":"percentage_risk", "params":{"risk_pct":0.02, "min_lots":0.01, "max_lots":1}},

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
        stride: int = 1,
        eval_metrics: dict = {}
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
        hot_reload=False,
        
    )

    split_name = os.path.basename(seq_data_path)
    files = sorted(glob.glob(str(seq_data_path) + f"/{split_name}_*.gz"))
    load_data = tf.data.TFRecordDataset(files, compression_type=config.tf_record_compression_type, num_parallel_reads=tf.data.AUTOTUNE)
    xgboost_model = model.xgb_model
    model = model.get_serving_signature()
    
    batch_dataset = load_data.batch(128).take(-1)

    df_dict = {
        'timestamp': [],
        'signal_type': [],
        'price': [],
    }
    print("metrics: ",eval_metrics)
    buy_valid = eval_metrics["is_buy_valid"]
    sell_valid= eval_metrics["is_sell_valid"]
    is_xgb = True if xgboost_model else None
    valid_classes = []
    if buy_valid:
        valid_classes.append(0)
    if sell_valid:
        valid_classes.append(1)
  
    valid_classes = tf.constant(valid_classes)
    for batch in batch_dataset:
        predictions = model(batch)['output']
        if is_xgb:
            predictions = xgboost_model.predict(predictions)

        batch_data = tf.io.parse_example(batch, features={
            'time': tf.io.FixedLenFeature([sequence_length], tf.int64),
            'close': tf.io.FixedLenFeature([sequence_length], tf.float32)
        })
        batch_times = batch_data['time'][:, -1]
        batch_prices = batch_data['close'][:, -1]
        pred_types = tf.where(predictions == 0, "BUY", tf.where(predictions == 1, "SELL", "HOLD"))
        mask = tf.reduce_any(tf.equal(tf.expand_dims(predictions, -1), valid_classes), axis=-1)
        valid_signals = tf.reshape(tf.where(mask), [-1])
        # valid_signals = tf.reshape(tf.where(tf.not_equal(predictions, 2)), [-1])
        df_dict['signal_type'].extend([t.decode('utf-8') for t in tf.gather(pred_types, valid_signals).numpy()])
        df_dict['timestamp'].extend(tf.gather(batch_times, valid_signals).numpy())
        df_dict['price'].extend(tf.gather(batch_prices, valid_signals).numpy())

    df = pd.DataFrame(df_dict)
    print("total signals: ", len(df))
    if len(df)<2:
        print("No signal found in test data.")
        return df
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
            "data_source": os.getenv("DATA_SOURCE"),
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