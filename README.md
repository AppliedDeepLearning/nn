A neural network library with a high-level API built on top of TensorFlow. Most of the components are either thin wrappers or aliases for the TensorFlow components and can be freely mixed with native TensorFlow code.


<!-- TOC depthFrom:2 depthTo:3 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Installation](#installation)
- [Getting Started](#getting-started)
	- [Example: CNN MNIST Classifier](#example-cnn-mnist-classifier)
- [Documentation](#documentation)
	- [Model Function](#model-function)
	- [Activations](#activations)
	- [Layers](#layers)
	- [Losses](#losses)
	- [Metrics](#metrics)
	- [Models](#models)
	- [Optimizers](#optimizers)
- [License](#license)

<!-- /TOC -->


## Installation

```sh
pip install nn
```

It is recommended to use a [virtual environment].

## Getting Started

```py
import nn
from nn.layers import Dense

# Define the network architecture - layers, number of neurons, activations etc.
def network(inputs):
    hidden = Dense(50, activation=nn.relu)(inputs)
    outputs = Dense(10, activation=nn.sigmoid)(hidden)
    return outputs

# Configure the model parameters - loss, optimizer, evaluation metrics etc.
model = nn.models.Classifier(
    network,
    optimizer=nn.optimizers.SGD(0.01))

# Train the model using training data
model.train(x_train, y_train, num_epochs=30, batch_size=100)

# Evaluate the model performance on test or validation data
print(model.evaluate(x_test, y_test))

# Use the model to make predictions for new data
print(model.predict(x_test))
```

More configuration options are available:

```py
model = nn.models.Classifier(
    network,
    loss=nn.losses.softmax_cross_entropy,
    optimizer=nn.optimizers.SGD(0.01),
    metrics=[nn.metrics.accuracy],
    model_dir='/tmp/my_model')
```

If a network contains only a sequence of layers, it can be written more compactly as:

```py
from nn.layers import Sequence

def network(inputs): return Sequence([
    Dense(50, activation=nn.relu),
    Dense(10, activation=nn.sigmoid),
])(inputs)
```

### Example: CNN MNIST Classifier

This example is based on the [MNIST example] of TensorFlow:

```py
import nn
from nn.layers import Sequence, Reshape, Flatten
from nn.layers import Conv2D, MaxPooling2D, Dense, Dropout

def network(inputs, mode): return Sequence([
    Reshape([28, 28, 1]),
    Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=nn.relu),
    MaxPooling2D(pool_size=[2, 2], strides=2),
    Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=nn.relu),
    MaxPooling2D(pool_size=[2, 2], strides=2),
    Flatten(),
    Dense(1024, activation=nn.relu),
    Dropout(rate=0.4, mode=mode),
    Dense(10),
])(inputs)

model = nn.models.Classifier(
    network,
    loss=nn.losses.sparse_softmax_cross_entropy,
    optimizer=nn.optimizers.GradientDescent(0.001),
    metrics=[nn.metrics.accuracy])
```

> **Note:** `mode` parameter tells whether the model is used for training, evaluation or prediction and should be passed to `Dropout` layers.


## Documentation

### Model Function

To have more control over the model, you may define the model using a function:

```py
import nn
from nn.layers import Sequence, Dense
import tensorflow as tf

def model(features, labels, mode):
    # Define the network architecture
    outputs = Sequence([
        Dense(50, activation=nn.relu),
        Dense(10, activation=nn.sigmoid),
    ])(features)
    predictions = tf.argmax(outputs, axis=1)

    # Configure the model parameters
    loss = nn.losses.softmax_cross_entropy(labels, outputs)
    optimizer = nn.optimizers.GradientDescent(0.01)
    accuracy = nn.metrics.accuracy(labels, predictions)

    return nn.models.spec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        optimizer=optimizer,
        metrics={'accuracy': accuracy})

# Create the model using model function
model = nn.models.Model(model)

# Train the model
model.train(x_train, y_train, num_epochs=30, batch_size=100)
```

`nn.models.spec` helper function creates a [`tf.estimator.EstimatorSpec`][tf.estimator.EstimatorSpec] object using `mode` and other parameters.

### Activations

Aliases for [TensorFlow activation functions] are available at the package root. For example, `tf.nn.relu` is available as `nn.relu`.

### Layers

Most of the `nn.layers` are aliases for [`tf.layers`][tf.layers] except:

- Dropout
- Reshape
- Sequence

`nn.layers.Dropout` accepts `mode` as a parameter whereas `tf.layers.Dropout` accepts `training` as a parameter and their signatures are also different:

```py
nn.layers.Dropout(rate=0.4, mode=mode)(inputs)
tf.layers.Dropout(rate=0.4)(inputs, training=(mode == tf.estimator.ModeKeys.TRAIN))
```

### Losses

`nn.losses` are aliases for [`tf.losses`][tf.losses].

### Metrics

Most of the `nn.metrics` are aliases for [`tf.metrics`][tf.metrics] except:

- accuracy

`nn.metrics.accuracy` first converts `labels` and `predictions` one-hot tensors to dense tensors and then applies `tf.metrics.accuracy`.

### Models

`nn.models` are wrappers for [`tf.estimator.Estimator`][tf.estimator.Estimator] and [`tf.estimator.EstimatorSpec`][tf.estimator.EstimatorSpec].

- Classifier
- Regressor
- Model
- spec

### Optimizers

Most of the `nn.optimizers` are aliases for [TensorFlow optimizers] except:

- SGD

The word `Optimizer` is excluded from all optimizer names. For example, `GradientDescentOptimizer` of TensorFlow is available as `nn.optimizers.GradientDescent`.


## License

[MIT][license]


[license]: /LICENSE
[virtual environment]: https://docs.python.org/3/library/venv.html
[MNIST example]: https://www.tensorflow.org/tutorials/layers#building_the_cnn_mnist_classifier
[TensorFlow activation functions]: https://www.tensorflow.org/api_guides/python/nn#Activation_Functions
[tf.layers]: https://www.tensorflow.org/api_docs/python/tf/layers
[tf.losses]: https://www.tensorflow.org/api_docs/python/tf/losses
[tf.metrics]: https://www.tensorflow.org/api_docs/python/tf/metrics
[tf.estimator.Estimator]: https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
[tf.estimator.EstimatorSpec]: https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec
[TensorFlow optimizers]: https://www.tensorflow.org/api_guides/python/train#Optimizers
