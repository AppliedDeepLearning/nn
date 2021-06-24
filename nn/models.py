import shutil
import tensorflow as tf

from . import losses
from . import metrics
from . import modes
from .train import TrainingHook, NotTrainingHook, optimizer as create_optimizer
from .utils import call_fn, logger, dataset


class Model():

    def __init__(self, model_fn, **kwargs):
        self._model_fn = model_fn
        self._estimator_kwargs = kwargs
        self._estimator = None
        self._hooks = self._get_hooks()

    def compile(self, **kwargs):
        self._estimator_kwargs.update(kwargs)
        self._estimator = self._create_estimator(model_fn=self._model_fn, **self._estimator_kwargs)

    def train(self, x, y=None, epochs=1, batch_size=128, shuffle=True, **kwargs):
        input_fn = dataset(x=x,
                           y=y,
                           epochs=epochs,
                           batch_size=batch_size,
                           shuffle=shuffle)
        self._update_kwargs(modes.TRAIN, kwargs)
        self.estimator.train(input_fn=input_fn, **kwargs)

    def evaluate(self, x, y=None, batch_size=128, **kwargs):
        input_fn = dataset(x=x,
                           y=y,
                           batch_size=batch_size,
                           shuffle=False)
        self._update_kwargs(modes.EVAL, kwargs)
        return self.estimator.evaluate(input_fn=input_fn, **kwargs)

    def predict(self, x, batch_size=128, **kwargs):
        input_fn = dataset(x=x,
                           batch_size=batch_size,
                           shuffle=False)
        self._update_kwargs(modes.PREDICT, kwargs)
        return self.estimator.predict(input_fn=input_fn, **kwargs)

    __call__ = predict

    @property
    def estimator(self):
        if self._estimator is None:
            self.compile()
        return self._estimator

    def _create_estimator(self, model_fn, directory=None, model_dir=None, params=None, **kwargs):
        defaults = self._defaults()
        if model_dir is None:
            model_dir = directory
        if model_dir is None:
            model_dir = defaults.get('model_dir')
            shutil.rmtree(model_dir, ignore_errors=True)
            logger.warn('Using temporary folder as model directory: {}'.format(model_dir))
        params = defaults.get('params', {}) if params is None else params
        model_fn = self._create_model_fn(model_fn)
        return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=params, **kwargs)

    def _defaults(self):
        return {
            'model_dir': '/tmp/nn_model_dir',
        }

    def _create_model_fn(self, model_fn):

        def fn(features, labels, mode, params, config):
            if list(features.keys()) == ['x']:
                features = features['x']

            ret = call_fn(model_fn, features, labels=labels, mode=mode, params=params, config=config)
            if not isinstance(ret, tf.estimator.EstimatorSpec):
                if not isinstance(ret, dict):
                    ret = dict(outputs=ret)
                ret = spec(mode=mode, labels=labels, **ret)
            return ret

        return fn

    def _get_hooks(self):
        return {
            modes.TRAIN: [TrainingHook()],
            modes.EVAL: [NotTrainingHook()],
            modes.PREDICT: [NotTrainingHook()],
        }

    def _update_kwargs(self, mode, kwargs):
        hooks = self._hooks.get(mode, []) + kwargs.get('hooks', [])
        if hooks:
            kwargs['hooks'] = hooks


def model(model_fn=None, **kwargs):

    def decorator(model_fn):
        return Model(model_fn, **kwargs)

    return decorator(model_fn) if model_fn is not None else decorator


def spec(mode, labels=None, outputs=None, predictions=None, loss=None, optimizer=None, metrics=None, **kwargs):
    if predictions is None:
        predictions = outputs
    if mode == modes.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, **kwargs)

    loss = create_loss(loss, labels, outputs)

    if mode == modes.TRAIN:
        train_op = create_train_op(optimizer, loss)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, **kwargs)

    metrics = create_metric_ops(metrics, labels, outputs)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics, **kwargs)


def create_loss(loss, labels, outputs):
    if isinstance(loss, tf.Tensor):
        return loss
    if isinstance(loss, str):
        loss = getattr(losses, loss)
    return loss(labels, outputs)


def create_train_op(optimizer, loss):
    if isinstance(optimizer, tf.Operation):
        return optimizer
    global_step = tf.train.get_global_step()
    if isinstance(optimizer, tuple):
        optimizer = create_optimizer(*optimizer)
    if not isinstance(optimizer, tf.train.Optimizer) and callable(optimizer):
        return call_fn(optimizer, loss, global_step=global_step)
    return optimizer.minimize(loss=loss, global_step=global_step)


def create_metric_ops(metrics, labels, outputs):
    metrics = metrics or {}
    if isinstance(metrics, list):
        metrics = {getattr(metric, '__name__', metric): metric for metric in metrics}
    return {name: create_metric_op(metric, labels, outputs) for name, metric in metrics.items()}


def create_metric_op(metric, labels, outputs):
    if isinstance(metric, tf.Operation):
        return metric
    if isinstance(metric, str):
        metric = getattr(metrics, metric)
    return metric(labels, outputs)
