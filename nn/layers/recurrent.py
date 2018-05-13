from tensorflow.python.ops.rnn_cell_impl import *
import tensorflow as tf

from .. import sequence


class RNN(tf.layers.Layer):

    def __init__(self, cell, return_sequences=False, return_state=False, **kwargs):
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        super(RNN, self).__init__(**kwargs)

    def call(self, inputs, sequence_length=None, dtype=tf.float32, **kwargs):
        outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=sequence_length, dtype=dtype, **kwargs)
        if not self.return_sequences:
            if sequence_length is not None:
                outputs = sequence.batch_gather_nd(outputs, sequence_length - 1)
            else:
                outputs = outputs[:, -1, :]
        if self.return_state:
            return outputs, state
        else:
            return outputs
