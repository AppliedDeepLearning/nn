import tensorflow as tf

GradientDescent = tf.train.GradientDescentOptimizer
Adadelta = tf.train.AdadeltaOptimizer
Adagrad = tf.train.AdagradOptimizer
AdagradDA = tf.train.AdagradDAOptimizer
Momentum = tf.train.MomentumOptimizer
Adam = tf.train.AdamOptimizer
Ftrl = tf.train.FtrlOptimizer
ProximalGradientDescent = tf.train.ProximalGradientDescentOptimizer
ProximalAdagrad = tf.train.ProximalAdagradOptimizer
RMSProp = tf.train.RMSPropOptimizer


def SGD(
    learning_rate,
    momentum=0.0,
    decay_rate=None,
    decay_steps=5000,
    staircase=False,
    global_step=None,
    **keywords
):

    def optimizer():
        lr = learning_rate
        if decay_rate is not None:
            lr = tf.train.exponential_decay(
                learning_rate=float(learning_rate),
                global_step=tf.train.get_global_step() if global_step is None else global_step,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=staircase,
            )
        return Momentum(lr, momentum=momentum, **keywords)

    return optimizer
