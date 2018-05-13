import tensorflow as tf


def length(x, pad=0):
    valid = tf.not_equal(x, pad)
    valid = tf.cast(valid, tf.int32)
    return tf.reduce_sum(valid, axis=1)


# See https://stackoverflow.com/a/43298689
def batch_gather_nd(params, indices, axis=1):
    batch_range = tf.range(tf.shape(params)[0])
    indices = tf.stack([batch_range, indices], axis=axis)
    return tf.gather_nd(params, indices)
