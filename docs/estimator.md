# Estimator

To have more control, you may configure the model inside a function using `Estimator` class:

```py
def model_fn(features, labels, mode):
    # Define the network architecture
    hidden = nn.Dense(units=64, activation='relu')(features)
    outputs = nn.Dense(units=10)(hidden)
    predictions = tf.argmax(outputs, axis=1)

    # In prediction mode, simply return predictions without configuring learning process
    if mode == nn.PREDICT:
        return predictions

    # Configure the learning process for training and evaluation modes
    loss = nn.losses.softmax_cross_entropy(labels, outputs)
    optimizer = nn.optimizers.GradientDescent(0.001)
    accuracy = nn.metrics.accuracy(labels, predictions)
    return dict(loss=loss,
                optimizer=optimizer,
                metrics={'accuracy': accuracy})
```

`mode` parameter specifies whether the model is used for training, evaluation or prediction. Unlike [`nn.Model`][nn.Model], here `loss` should be an operation (not a function or function name) and `metrics` should be a `dict` (not a `list`) of names and operations (not functions or function names).

Create a model using the model function:

```py
model = nn.Estimator(model_fn)
```

Save model parameters, training progress etc. by specifying a model directory:

```py
model = nn.Estimator(model_fn, model_dir='/tmp/my_model')
```

Rest of the API is same as [`nn.Model`][nn.Model].

Also you may use a neural network function that is created for [`nn.Model`][nn.Model]:

```py
def network(inputs):
    ...
    return outputs

model = nn.Model(network, ...)

def model_fn(features, labels):
    outputs = network(features)
    ...

model = nn.Estimator(model_fn)
```


[nn.Model]: ../model/
