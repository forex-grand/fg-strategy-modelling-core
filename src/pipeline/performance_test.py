import tensorflow as tf 
import pandas as pd
import tempfile
import asyncio
import requests
from src.models_architecture.base_model import BaseModel
from src.settings import Settings
from src.pipeline.generate_train_data import GenerateTrainData

trade_managers = [
    ###fixed stop loss and take profit points
    {"type":"fixed_sl_tp","params":{"stop_loss_points":50,"take_profit_points":100}},
    {"type":"fixed_sl_tp","params":{"stop_loss_points":100,"take_profit_points":200}},
    {"type":"fixed_sl_tp","params":{"stop_loss_points":100,"take_profit_points":100}},
    {"type":"fixed_sl_tp","params":{"stop_loss_points":200,"take_profit_points":200}},
    {"type":"fixed_sl_tp","params":{"stop_loss_points":200,"take_profit_points":100}},

    ###RISK REWARD BASED
    # {"type":"risk_reward", "params":{"rr_ratio":1.0, 
    #                         "sl_calculator":{"type":"fixed","sl_point":200}}},
    # {"type":"fixed", "params":{"rr_ratio":2.0, 
    #             "sl_calculator":{"type":"atr","atr_period":14,"atr_multiplier":1.5,"atr_timeframe":"M5"}}},
    # {"type":"risk_reward", "params":{"rr_ratio":1.0,
    #             "sl_calculator":{"type":"stdev", "stdev_period":20,"stdev_multiplier":2.0,"stdev_timeframe":"M5"}}},
    # {"type":"fixed", "params":{"rr_ratio":2.0, 
    #             "sl_calculator":{"type":"swing","swing_bars":21,"swing_timeframe":"M5"}}},
    # ###SL DISTANCE TRIGGER
    # {"type":"sl_distance_trigger", "params":{"sl_ratio":2.0}},
    # ###TIME STOP BASED
    # {"type":"time_stop", "params": {"max_duration_minutes":240}},
    ]


def test_model_live_performance(
        model:BaseModel, 
        symbol: str,
        group: str,
        sequence_length: int,
        config: Settings,
        model_id: str,
    ):
    data_gen = GenerateTrainData(eval_base_bucket=config.eval_bucket_name, 
                                 train_base_bucket=config.eval_bucket_name)
    seq_data1,_ = data_gen.load_data(symbol_pair=symbol, instrument_group=group, 
                                  sequence_length=sequence_length, stride=1000, hot_reload=False)
    
    load_data = tf.data.TFRecordDataset(seq_data1, compression_type=config.tf_record_compression_type)
    # load_data = data_ser.take(20)
    model = model.get_serving_signature()
    batch_dataset = load_data.take(100).batch(20)

    df_dict = {
        'timestamp':[],
        'signal_type':[],
    }
    for batch in batch_dataset:
        predictions = model(batch)['output']
        batch_times = tf.io.parse_example(batch, features={'time':tf.io.FixedLenFeature([sequence_length], tf.int64)})['time'][-1]
        pred_types = tf.where(predictions==0,"BUY",tf.where(predictions==1, "SELL","HOLD"))
        valid_signals = tf.squeeze(tf.where(tf.not_equal(predictions, 2)))
        df_dict['signal_type'].extend([t.decode('utf-8') for t in tf.gather(pred_types, valid_signals).numpy()])
        df_dict['timestamp'].extend(tf.gather(batch_times, valid_signals).numpy())

    df = pd.DataFrame(df_dict)
    df['price'] = 0.0
    # print(df.head())

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        signals = None
        df.to_csv(temp_file.name, index=False)
        with open(temp_file.name, 'rb') as f:
            signals = requests.post(f'{config.performance_base_url}/engine/signals/upload', data={'timestamp_unit':'auto'}, files={'file':f})

        performance_test_request = {
            "strategy":{
            'source': 'csv',
            'type': '',
            'params': {},
            },
            'uploaded_signals': signals.json()['signals'],
            "symbols":[symbol],
            "trade_managers":trade_managers,
            "lotsizers":[{"type":"fixed_lot_size","params":{"fixed_lots":1}}],
            "data_source":"metaquotes",
            "name":f"{model_id}_{symbol}"
        }

        ###send request to backend
        response = requests.post(f'{config.performance_base_url}/performance-tests', json=performance_test_request)
        print("Test Job Id: ",response.json()['job_id']," Status: ",response.status_code)