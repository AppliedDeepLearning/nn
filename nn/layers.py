import tensorflow as tf
from tensorflow.python.layers.layers import *


def Dropout(mode, **keywords):
    return lambda inputs: dropout(inputs, training=(mode == tf.estimator.ModeKeys.TRAIN), **keywords)


def Reshape(shape, **keywords):
    shape = [-1] + shape  # preserve the batch axis (axis 0)
    return lambda tensor: tf.reshape(tensor, shape, **keywords)


def Sequence(layers):

    def layer(x):
        for l in layers:
            x = l(x)
        return x

    return layer


def concatenate(tensors):
    return tf.concat(tensors, 1)
