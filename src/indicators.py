import tensorflow as tf
import keras 

##MOVING AVERAGE
def tf_ma(
    tensors: tf.Tensor,
    period=12
  ):
  """Computes moving average on each tensor. returns a shape of [batch, sequence_length-period+1]"""
  if len(tensors.shape)<2:
    tensors = tf.expand_dims(tensors, 0)

  return tf.reduce_mean(tf.signal.frame(tensors, period, 1), axis=-1)

##SLOPE
def tf_slope(
    tensors: tf.Tensor,
    period=14
  ):
  """Computes slope, intercept on individual tensor.
    The slope is computed on the last periods of data tensor[-period:]
  """
  if len(tensors.shape)<2:
    tensors = tf.expand_dims(tensors, 0)

  seq_length = period if period else tensors.shape[-1]
  x_values = tf.range(0,seq_length, dtype=tf.float32)
  x_bar = tf.reduce_mean(x_values)
  x_xm = tf.subtract(x_values, x_bar)
  y_bar = tf.reduce_mean(tensors[:,-seq_length:], axis=-1)
  y_ym = tf.subtract(tensors[:,-seq_length:], tf.expand_dims(y_bar, -1))

  numerator = tf.reduce_sum(x_xm*y_ym, axis=-1)
  slope = numerator/tf.reduce_sum(tf.square(x_xm))
  intercept = y_bar - (slope*x_bar)
  return slope, intercept

##ATR
def tf_atr(
    open_tensor: tf.Tensor,
    high_tensor: tf.Tensor,
    close_tensor: tf.Tensor,
    low_tensor: tf.Tensor,
    period=14
  ):
  ##RETURNS A TENSOR OF SHAPE [BATCH, ORIGINAL_SHAPE-PERIOD]
  tr = tf.reduce_max(tf.stack([tf.subtract(high_tensor, low_tensor)[:,1:], tf.math.abs(tf.subtract(high_tensor[:,1:], close_tensor[:,:-1])), tf.math.abs(tf.subtract(low_tensor[:,1:], close_tensor[:,:-1]))]), axis=0)
  avg_vals = tf.reduce_mean(tr[:,:period], axis=-1)
  atr_values = tf.TensorArray(size=tr.shape[-1]-period+1, dtype=tf.float32, clear_after_read=False)
  atr_values = atr_values.write(0, avg_vals)

  def calculate_atr(idx, atr_values):
    prev_atr = atr_values.read(idx-1)
    curr_atr = tf.math.divide(tf.add(tf.math.multiply(prev_atr,period-1), tr[:,period+idx-1]), period)
    atr_values = atr_values.write(idx, curr_atr)
    return idx+1, atr_values

  iter_counts = tr.shape[-1] - period + 1
  _, updated_atr_values = tf.while_loop(
      lambda i,*args: i<iter_counts,
      calculate_atr,
      [1, atr_values]
  )

  atr_values = tf.transpose(updated_atr_values.stack())
  return atr_values

##RSI
def tf_rsi(tensor: tf.Tensor, period=14):
  ##RETURNS A TENSOR OF SHAPE [BATCH, ORIGINAL_SHAPE-PERIOD+1]
  diff_i      = tf.subtract(tensor[:,1:period], tensor[:,:period-1])
  pos_i       = tf.where(tf.greater(diff_i, 0.0), diff_i, 0.0)
  pos_mean  = tf.reduce_mean(pos_i, axis=-1)
  neg_i       = tf.where(tf.less(diff_i, 0.0), tf.abs(diff_i), 0.0)
  neg_mean  = tf.reduce_mean(neg_i, axis=-1)

  val = tf.where(
      tf.greater(neg_mean, 0.0),
      tf.math.subtract(100.0, tf.math.divide(100.0, tf.math.add(1.0, tf.math.divide(pos_mean, neg_mean)))),
      tf.where(
          tf.greater(pos_mean, 0.0),
          100.0,
          50.0
      )
  )

  fill_length = tensor.shape[-1] - period +1
  pos_buffers = tf.TensorArray(size=fill_length, dtype=tf.float32, clear_after_read=False)
  pos_buffers = pos_buffers.write(0, pos_mean)
  neg_buffers = tf.TensorArray(size=fill_length, dtype=tf.float32, clear_after_read=False)
  neg_buffers = neg_buffers.write(0, neg_mean)
  rsi_vals    = tf.TensorArray(size=fill_length, dtype=tf.float32, clear_after_read=False)
  rsi_vals    = rsi_vals.write(0, val)

  diff  =   tf.subtract(tensor[:,period-1:], tensor[:,period-2:-1])
  pos   =   tf.where(tf.greater(diff, 0.0), diff, 0.0)
  neg   =   tf.where(tf.less(diff, 0.0), tf.abs(diff), 0.0)

  def calculate_rsi(idx, pos_buffer, neg_buffer, rsi_buffer):
    pos_buffer_val = tf.math.divide(tf.add(tf.math.multiply(pos_buffer.read(idx-1), period-1), pos[:,idx]), period)
    neg_buffer_val = tf.math.divide(tf.add(tf.math.multiply(neg_buffer.read(idx-1), period-1), neg[:,idx]), period)

    rsi_val = tf.where(
        tf.greater(neg_buffer_val, 0.0),
        tf.subtract(tf.constant(100.0, tf.float32), tf.math.divide(tf.constant(100.0, tf.float32), tf.math.add(tf.constant(1.0, tf.float32), tf.math.divide(pos_buffer_val, neg_buffer_val)))),
        tf.where(
            tf.greater(pos_buffer_val, 0.0),
            tf.constant(100.0, tf.float32),
            tf.constant(50.0, tf.float32)
            )
    )
    pos_buffer = pos_buffer.write(idx, pos_buffer_val)
    neg_buffer = neg_buffer.write(idx, neg_buffer_val)
    rsi_buffer = rsi_buffer.write(idx, rsi_val)
    return idx+1, pos_buffer, neg_buffer, rsi_buffer

  _, pos_buffers, neg_buffer, rsi_buffer = tf.while_loop(
      lambda i,*args: i<fill_length,
      calculate_rsi,
      [1, pos_buffers, neg_buffers, rsi_vals]
  )
  return tf.transpose(rsi_buffer.stack())

##STANDARD DEVIATION
def tf_stdev(price_tensor: tf.Tensor, period=14):
  ##RETURNS A TENSOR OF SHAPE [BATCH, ORIGINAL_SHAPE-PERIOD+1]
  ma = get_ma(price_tensor, period)
  deviations = tf.TensorArray(tf.float32, ma.shape[-1])
  def calculate_dev(idx, deviation_buffer):
    deviation = tf.math.sqrt(tf.reduce_sum(tf.pow(tf.subtract(price_tensor[:,idx:idx+period-1], tf.expand_dims(ma[:,idx], -1)), 2.0), -1))
    deviation_buffer = deviation_buffer.write(idx, deviation)
    return idx+1, deviation_buffer

  _, deviation_buff = tf.while_loop(
      lambda i,*args: i<ma.shape[-1],
      calculate_dev,
      [0, deviations]
  )

  return tf.transpose(deviation_buff.stack())

##BOLLINGER BANDS
def tf_bb(price_tensor: tf.Tensor, period=14, deviation=2.0):
  ##RETURNS A TENSOR OF SHAPE [BATCH, ORIGINAL_VALUES_SHAPE-PERIOD+1]
  ma = get_ma(price_tensor, period)
  upper_band_buffer = tf.TensorArray(tf.float32, ma.shape[-1])
  lower_band_buffer = tf.TensorArray(tf.float32, ma.shape[-1])

  def calculate_bands(idx, upper_band_buffer, lower_band_buffer):
    stdev = tf.math.sqrt(tf.reduce_sum(tf.pow(tf.subtract(price_tensor[:,idx:idx+period-1], tf.expand_dims(ma[:,idx], -1)), 2.0), -1))
    upper_band_buffer = upper_band_buffer.write(idx, tf.add(ma[:,idx], tf.math.multiply_no_nan(stdev, deviation)))
    lower_band_buffer = lower_band_buffer.write(idx, tf.subtract(ma[:,idx], tf.math.multiply_no_nan(stdev, deviation)))

    return idx+1, upper_band_buffer, lower_band_buffer

  _, upper_buffer, lower_buffer = tf.while_loop(
      lambda i,*args: i<ma.shape[-1],
      calculate_bands,
      [0, upper_band_buffer, lower_band_buffer]
  )

  return ma, tf.transpose(upper_buffer.stack()), tf.transpose(lower_buffer.stack())

def tf_german_klass_volatility(open_tensor: tf.Tensor, high_tensor: tf.Tensor, close_tensor: tf.Tensor, low_tensor: tf.Tensor, period=14):
  log_hl = tf.math.log(tf.math.divide_no_nan(high_tensor, low_tensor))
  log_co = tf.math.log(tf.math.divide_no_nan(close_tensor, open_tensor))

  gk = tf.subtract(tf.multiply(tf.square(log_hl), 0.5), tf.multiply(tf.subtract(tf.multiply(2, tf.math.log(2.0)), 1.0), tf.square(log_co)))
  mean = tf.sqrt(tf.reduce_mean(tf.signal.frame(gk, period, 1), -1), -1)
  anualized = tf.multiply(mean, tf.sqrt(tf.cast(period, tf.float32)))
  return anualized

def tf_wick_bar_range_ratio(open_tensor: tf.Tensor, high_tensor: tf.Tensor, close_tensor: tf.Tensor, low_tensor: tf.Tensor):
  bar_range = tf.abs(tf.subtract(open_tensor, close_tensor))
  wick_range = tf.subtract(high_tensor, low_tensor)
  ratio = tf.math.divide_no_nan(bar_range, wick_range)
  return ratio

def tf_feature_clip_values(tensor, lower_quantile=0.01, upper_quantile=0.99):
  q1 = keras.ops.quantile(tensor, lower_quantile, axis=-1)
  q2 = keras.ops.quantile(tensor, upper_quantile, axis=-1)

  clipped_values = tf.clip_by_value(tensor, tf.expand_dims(q1, -1), tf.expand_dims(q2, -1))

  normalizer = keras.layers.Normalization(axis=0)
  normalizer.adapt(clipped_values)
  return normalizer(clipped_values)[:,-1]