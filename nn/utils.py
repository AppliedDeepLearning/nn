import tensorflow as tf
import logging

logger = logging.getLogger('nn')
logger.setLevel(logging.INFO)


def to_dense(x):
    return tf.argmax(x, axis=1) if len(x.shape) > 1 and x.shape[1] > 1 else x
