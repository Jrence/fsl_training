import numpy as np
from tensorflow.keras import layers
import tensorflow as tf


# 1. ULITIN ANG CUSTOM LAYER DEFINITION
@tf.keras.utils.register_keras_serializable()
class TFLiteGraphConv(layers.Layer):

  def __init__(self, out_channels, adj_matrix, **kwargs):
    super().__init__(**kwargs)
    self.out_channels = out_channels
    self.adj_matrix = adj_matrix
    self.A = tf.constant(adj_matrix, dtype=tf.float32)

  def build(self, input_shape):
    self.dense = layers.Dense(self.out_channels, use_bias=False)
    super().build(input_shape)

  def get_config(self):
    config = super().get_config()
    config.update({
        "out_channels": self.out_channels,
        "adj_matrix": (
            self.adj_matrix.tolist()
            if isinstance(self.adj_matrix, np.ndarray)
            else self.adj_matrix
        ),
    })
    return config

  def call(self, inputs):
    x_weight = self.dense(inputs)
    shape = tf.shape(inputs)
    batch_size, frames = shape[0], shape[1]
    x_reshaped = tf.reshape(x_weight, [-1, 75, self.out_channels])
    x_graph = tf.matmul(self.A, x_reshaped)
    return tf.reshape(x_graph, [batch_size, frames, 75, self.out_channels])


# 2. LOAD ANG KERAS MODEL
print("Loading Keras model...")
model = tf.keras.models.load_model(
    "fsl_model.keras",
    custom_objects={"TFLiteGraphConv": TFLiteGraphConv},
    compile=False,
)

# 3. I-CONVERT PATUNGONG TFLITE
print("Converting model to TFLite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Sinisiguradong suportado ang custom GraphConv operations sa mobile
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]

tflite_model = converter.convert()

# 4. ISAVE ANG .TFLITE FILE
tflite_filename = "fsl_model.tflite"
with open(tflite_filename, "wb") as f:
  f.write(tflite_model)

print(f"Tagumpay! Na-save ang TFLite model sa: {tflite_filename}")