import tensorflow as tf
from tensorflow.python.ops.metrics import *

from .utils import to_dense


def accuracy(labels, predictions, **keywords):
    labels = to_dense(labels)
    predictions = to_dense(predictions)
    return tf.metrics.accuracy(labels, predictions, **keywords)
