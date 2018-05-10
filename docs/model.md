# Model

Creating a model involves two steps:

1. Defining the neural network (layers, number of units, activations)
1. Configuring the learning process (loss, optimizer, evaluation metrics)

Once a model is created it can used for training or making predictions.

## Defining Neural Network

A neural network is created as a function. Defining a neural network as a function allows it to be created multiple times with different parameters for training, evaluation and prediction. A neural network function should take a set of inputs and return a set of outputs:

```py
def network(inputs):
    # Compute outputs
    return outputs
```

It can also use the `mode` argument to handle different modes - `nn.TRAIN`, `nn.EVAL` and `nn.PREDICT`:

```py
def network(inputs, mode):
    # Construct the network based on mode
    return outputs
```


## Creating Model

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

If a `model_dir` is not specified, a temporary folder is used.


## Usage

Train the model using training data:

```py
model.train(x, y=None, epochs=1, batch_size=128, shuffle=True, **keywords)
```

Evaluate the model performance on test or validation data:

```py
loss_and_metrics = model.evaluate(x, y=None, batch_size=128, **keywords)
```

Use the model to make predictions for new data:

```py
predictions = model.predict(x, batch_size=128, **keywords)
# or call the model directly
predictions = model(x, batch_size=128, **keywords)
```

> `x` and `y` are expected to be NumPy arrays.


## Next Steps

- [Learn more about layers](./layers/)
- [Learn more about optimizers](./optimizers/)
