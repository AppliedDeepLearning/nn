import tensorflow as tf


class Embedding(tf.layers.Layer):

    def __init__(self, input_dim, output_dim,
                 embeddings_initializer=None,
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 dtype=tf.float32,
                 **kwargs):
        super(Embedding, self).__init__(dtype=dtype, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint

    def build(self, input_shape):
        self.embeddings = self.add_variable(
            name='embeddings',
            shape=[self.input_dim, self.output_dim],
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype,
            trainable=True)
        self.built = True

    def call(self, inputs):
        if inputs.dtype != 'int32':
            inputs = tf.cast(inputs, 'int32')
        return tf.nn.embedding_lookup(self.embeddings, inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape).concatenate(self.output_dim)
