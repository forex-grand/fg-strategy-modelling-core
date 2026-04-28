import tensorflow as tf 
import pandas as pd
import tempfile
import asyncio
from src.models_architecture.base_model import BaseModel
from src.settings import Settings
from src.pipeline.generate_train_data import GenerateTrainData

def test_model_live_performance(
        model:BaseModel, 
        symbol: str,
        group: str,
        sequence_length: int,
        config: Settings,
    ):
    data_gen = GenerateTrainData(eval_base_bucket=config.test_bucket_name, 
                                 train_base_bucket=config.test_bucket_name)
    seq_data1,_ = data_gen.load_data(symbol_pair=symbol, instrument_group=group, 
                                  sequence_length=sequence_length, stride=1, hot_reload=False)
    
    load_data = tf.data.TFRecordDataset(seq_data1, compression_type=config.tf_record_compression_type)
    # load_data = data_ser.take(20)
    model = model.get_serving_signature()
    batch_dataset = load_data.batch(20)

    df_dict = {
        'time':[],
        'type':[],
    }
    ##Collect all signals
    for batch in batch_dataset:
        predictions = model(batch)['output']
        batch_times = tf.io.parse_example(batch, features={'time':tf.io.FixedLenFeature([sequence_length], tf.int64)})['time'][-1]
        pred_types = tf.where(predictions==0,"BUY",tf.where(predictions==1, "SELL","HOLD"))
        valid_signals = tf.squeeze(tf.where(tf.not_equal(predictions, 2)))
        df_dict['type'].extend([t.decode('utf-8') for t in tf.gather(pred_types, valid_signals).numpy()])
        df_dict['time'].extend(tf.gather(batch_times, valid_signals).numpy())

    df = pd.DataFrame(df_dict)
    print(df.head())