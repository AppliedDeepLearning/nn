from tensorflow.python.layers.layers import *
import tensorflow as tf
from train import training as is_training

from . import activations


def _uses_activation(Layer, attributes=None):
    if attributes is None:
        attributes = ['activation']

    def init(self, *args, **kwargs):
        super(Layer, self).__init__(*args, **kwargs)
        for name in attributes:
            if hasattr(self, name):
                activation = getattr(self, name)
                if isinstance(activation, str):
                    activation = getattr(activations, activation)
                    setattr(self, name, activation)

    Layer.__init__ = init
    return Layer


def _uses_training(Layer):

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = is_training()
        return super(Layer, self).call(inputs, training=training, **kwargs)

    Layer.call = call
    return Layer


# Core

@_uses_activation
class Dense(tf.layers.Dense):
    pass


# Convolutional

@_uses_activation
class Conv1D(tf.layers.Conv1D):
    pass


@_uses_activation
class Conv2D(tf.layers.Conv2D):
    pass


@_uses_activation
class Conv2DTranspose(tf.layers.Conv2DTranspose):
    pass


@_uses_activation
class Conv3D(tf.layers.Conv3D):
    pass


@_uses_activation
class Conv3DTranspose(tf.layers.Conv3DTranspose):
    pass


@_uses_activation
class SeparableConv1D(tf.layers.SeparableConv1D):
    pass


@_uses_activation
class SeparableConv2D(tf.layers.SeparableConv2D):
    pass


# Training

@_uses_training
class Dropout(tf.layers.Dropout):
    pass


@_uses_training
class BatchNormalization(tf.layers.BatchNormalization):
    pass


# Others

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

__all__ = [
    'AveragePooling1D',
    'AveragePooling2D',
    'AveragePooling3D',
    'BatchNormalization',
    'Conv1D',
    'Conv2D',
    'Conv2DTranspose',
    'Conv3D',
    'Conv3DTranspose',
    'Dense',
    'Dropout',
    'Flatten',
    'InputSpec',
    'Layer',
    'MaxPooling1D',
    'MaxPooling2D',
    'MaxPooling3D',
    'SeparableConv1D',
    'SeparableConv2D',
]
