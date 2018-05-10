# NN: Python Neural Network Library

A neural network library built on top of TensorFlow for quickly building deep learning models.

## Installation

[Install TensorFlow]{:target="_blank"}:

```sh
pip install tensorflow
```

and run:

```sh
pip install nn
```

It is recommended to use a [virtual environment]{:target="_blank"}.


## Usage

Import the package:

```py
import nn
```

Define the network (layers, number of units, activations) as a function:

```py
def network(inputs):
    hidden = nn.Dense(units=64, activation='relu')(inputs)
    outputs = nn.Dense(units=10)(hidden)
    return outputs
```

Create a model by configuring its learning process (loss, optimizer, evaluation metrics):

```py
model = nn.Model(network,
                 loss='softmax_cross_entropy',
                 optimizer=('GradientDescent', 0.001),
                 metrics=['accuracy'])
```

Save/load model parameters, training progress etc. by specifying a model directory:

```py
model = nn.Model(network,
                 loss='softmax_cross_entropy',
                 optimizer=('GradientDescent', 0.001),
                 metrics=['accuracy'],
                 model_dir='/tmp/my_model')
```

Train the model using training data:

```py
model.train(x_train, y_train, epochs=30, batch_size=128)
```

Evaluate the model performance on test or validation data:

```py
loss_and_metrics = model.evaluate(x_test, y_test)
```

Use the model to make predictions for new data:

```py
predictions = model.predict(x)
# or call the model directly
predictions = model(x)
```


## Next Steps

- [Learn more about `nn.Model`](./model/)
- [Learn more about layers](./layers/)
- [Learn more about optimizers](./optimizers/)


[virtual environment]: https://docs.python.org/3/library/venv.html
[Install TensorFlow]: https://www.tensorflow.org/install/
