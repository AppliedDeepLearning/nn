from tensorflow.python.ops.rnn_cell_impl import *
import tensorflow as tf

from .. import sequence


def get_last_output(outputs, sequence_length):
    if sequence_length is not None:
        outputs = sequence.batch_gather_nd(outputs, sequence_length - 1)
    else:
        outputs = outputs[:, -1, :]
    return outputs


class RNN(tf.layers.Layer):

    def __init__(self, cell, cell_b=None, return_sequences=False, return_state=False, merge_mode='concat', **kwargs):
        self.cell = cell
        self.cell_b = cell_b
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.merge_mode = merge_mode
        self._trainable = True
        self._bidirectional = cell_b is not None
        super(RNN, self).__init__(**kwargs)

    def call(self, inputs, sequence_length=None, **kwargs):
        kwargs['inputs'] = inputs
        kwargs['sequence_length'] = sequence_length
        kwargs['dtype'] = kwargs.get('dtype', tf.float32)

        outputs, state = (tf.nn.bidirectional_dynamic_rnn(self.cell, self.cell_b, **kwargs)
                          if self._bidirectional else tf.nn.dynamic_rnn(self.cell, **kwargs))

        if not self.return_sequences:
            if self._bidirectional:
                outputs_fw, outputs_bw = outputs
                outputs_fw = get_last_output(outputs_fw, sequence_length)
                outputs_bw = outputs_bw[:, 0, :]
                outputs = outputs_fw, outputs_bw
            else:
                outputs = get_last_output(outputs, sequence_length)

        if self._bidirectional:
            if self.merge_mode is 'concat':
                outputs = tf.concat(outputs, axis=-1)

        if self.return_state:
            return outputs, state
        else:
            return outputs

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        self.cell.trainable = value
        if self.cell_b is not None:
            self.cell_b.trainable = value
