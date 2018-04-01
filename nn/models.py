import shutil
import tensorflow as tf

from .metrics import accuracy
from .utils import call_fn, logger, to_dense

_MODEL_DIR = '/tmp/nn_model_dir'


def create_optimizer(optimizer):
    if isinstance(optimizer, tf.train.Optimizer):
        return optimizer
    if callable(optimizer):
        return optimizer()
    return optimizer


def create_model_fn(network, loss_fn, optimizer, metrics, predict):
    if isinstance(metrics, list):
        metrics = {metric.__name__: metric for metric in metrics}

    def model_fn(features, labels, mode, params, config):
        features = features['x']
        outputs = call_fn(network, features, mode=mode, params=params, config=config)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = predict(outputs)
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = loss_fn(labels, outputs)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = create_optimizer(optimizer).minimize(
                loss=loss,
                global_step=tf.train.get_global_step(),
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metric_ops = {name: metric(labels, outputs) for name, metric in metrics.items()}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return model_fn


class Model:

    def __init__(self, network, optimizer, loss=None, metrics=None,
                 model_dir=None, params=None, predict=None, **keywords):
        defaults = self._defaults()
        loss = defaults.get('loss') if loss is None else loss
        metrics = defaults.get('metrics', {}) if metrics is None else metrics
        params = defaults.get('params', {}) if params is None else params
        predict = defaults.get('predict') if predict is None else predict
        if model_dir is None:
            shutil.rmtree(_MODEL_DIR, ignore_errors=True)
            model_dir = _MODEL_DIR
            logger.warn('Using temporary folder as model directory: {}'.format(model_dir))
        model_fn = create_model_fn(network, loss, optimizer, metrics, predict)
        self.model = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=params, **keywords)

    def _defaults(self):
        predict = lambda x: x
        loss = tf.losses.mean_squared_error
        return {
            'predict': predict,
            'loss': loss,
        }

    def train(self, features, labels, num_epochs=30, batch_size=100, shuffle=True, **keywords):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': features},
            y=labels,
            num_epochs=num_epochs,
            batch_size=batch_size,
            shuffle=shuffle
        )
        self.model.train(input_fn=input_fn, **keywords)

    def evaluate(self, features, labels, batch_size=100, **keywords):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': features},
            y=labels,
            batch_size=batch_size,
            shuffle=False
        )
        return self.model.evaluate(input_fn=input_fn, **keywords)

    def predict(self, features, batch_size=100, **keywords):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': features},
            batch_size=batch_size,
            shuffle=False
        )
        return self.model.predict(input_fn=input_fn, **keywords)


class Classifier(Model):

    def _defaults(self):
        predict = to_dense
        loss = tf.losses.softmax_cross_entropy
        metrics = [accuracy]
        return {
            **super()._defaults(),
            'predict': predict,
            'loss': loss,
            'metrics': metrics,
        }


class Regressor(Model):
    pass
