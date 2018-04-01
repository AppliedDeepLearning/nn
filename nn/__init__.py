import tensorflow as tf

# Shortcuts
from . import layers
from . import losses
from . import metrics
from . import models
from . import optimizers

# Activations
relu = tf.nn.relu
relu6 = tf.nn.relu6
crelu = tf.nn.crelu
elu = tf.nn.elu
selu = tf.nn.selu
softplus = tf.nn.softplus
softsign = tf.nn.softsign
dropout = tf.nn.dropout
bias_add = tf.nn.bias_add
softmax = tf.nn.softmax
sigmoid = tf.sigmoid
tanh = tf.tanh

# Modes
TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT
