import shutil
import tensorflow as tf

from .metrics import accuracy
from .utils import call_fn, logger, to_dense


def spec(mode, predictions, loss, optimizer, metrics=None):
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)


class Model():

    def __init__(self, model, **keywords):
        self.estimator = self._create_estimator(model_fn=model, **keywords)

    def train(self, features, labels, num_epochs=30, batch_size=100, shuffle=True, **keywords):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': features},
            y=labels,
            num_epochs=num_epochs,
            batch_size=batch_size,
            shuffle=shuffle)
        self.estimator.train(input_fn=input_fn, **keywords)

    def evaluate(self, features, labels, batch_size=100, **keywords):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': features},
            y=labels,
            batch_size=batch_size,
            shuffle=False)
        return self.estimator.evaluate(input_fn=input_fn, **keywords)

    def predict(self, features, batch_size=100, **keywords):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': features},
            batch_size=batch_size,
            shuffle=False)
        return self.estimator.predict(input_fn=input_fn, **keywords)

    def _create_estimator(self, model_fn, model_dir=None, params=None, **keywords):
        defaults = self._defaults()
        if model_dir is None:
            model_dir = defaults.get('model_dir')
            shutil.rmtree(model_dir, ignore_errors=True)
            logger.warn('Using temporary folder as model directory: {}'.format(model_dir))
        params = defaults.get('params', {}) if params is None else params
        model_fn = self._wrap_model_fn(model_fn)
        return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=params, **keywords)

    def _defaults(self):
        return {
            'model_dir': '/tmp/nn_model_dir',
        }

    def _wrap_model_fn(self, model_fn):

        def fn(features, labels, mode, params, config):
            features = features['x']
            return call_fn(model_fn, features, labels, mode=mode, params=params, config=config)

        return fn


def Classifier(*arguments, **keywords):
    defaults = {
        'metrics': [accuracy],
        'loss': tf.losses.softmax_cross_entropy,
        'predict': to_dense,
    }
    return create_model(defaults, *arguments, **keywords)


def Regressor(*arguments, **keywords):
    defaults = {
        'loss': tf.losses.mean_squared_error,
        'predict': lambda x: x,
    }
    return create_model(defaults, *arguments, **keywords)


def create_model(defaults, network, optimizer, loss=None, metrics=None, predict=None, **keywords):
    loss = defaults.get('loss') if loss is None else loss
    metrics = defaults.get('metrics', {}) if metrics is None else metrics
    predict = defaults.get('predict') if predict is None else predict
    if isinstance(metrics, list):
        metrics = {metric.__name__: metric for metric in metrics}
    model_fn = create_model_fn(network, loss, optimizer, metrics, predict)
    return Model(model_fn, **keywords)


def create_model_fn(network, loss_fn, optimizer_fn, metrics, predict):

    def model_fn(features, labels, mode, params, config):
        outputs = call_fn(network, features, mode=mode, params=params, config=config)
        predictions = predict(outputs)
        loss = loss_fn(labels, outputs)
        optimizer = create_optimizer(optimizer_fn)
        eval_metric_ops = {name: metric(labels, outputs) for name, metric in metrics.items()}
        return spec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            optimizer=optimizer,
            metrics=eval_metric_ops)

    return model_fn


def create_optimizer(optimizer):
    if isinstance(optimizer, tf.train.Optimizer):
        return optimizer
    if callable(optimizer):
        return optimizer()
    return optimizer
