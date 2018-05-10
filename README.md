A neural network library built on top of TensorFlow for quickly building deep learning models.


## Installation

[Install TensorFlow]:

```sh
pip install tensorflow
```

and run:

```sh
pip install nn
```

It is recommended to use a [virtual environment].


## Getting Started

```py
import nn

# Define the network (layers, number of units, activations) as a function:
def network(inputs):
    hidden = nn.Dense(units=64, activation='relu')(inputs)
    outputs = nn.Dense(units=10)(hidden)
    return outputs

# Create a model by configuring its learning process (loss, optimizer, evaluation metrics):
model = nn.Model(network,
                 loss='softmax_cross_entropy',
                 optimizer=('GradientDescent', 0.001),
                 metrics=['accuracy'])

# Train the model using training data:
model.train(x_train, y_train, epochs=30, batch_size=128)

# Evaluate the model performance on test or validation data:
loss_and_metrics = model.evaluate(x_test, y_test)

# Use the model to make predictions for new data:
predictions = model.predict(x)
# or call the model directly
predictions = model(x)
```


## Documentation

See [documentation](http://nn.applieddeeplearning.com/).


## License

[MIT][license]


[license]: /LICENSE
[virtual environment]: https://docs.python.org/3/library/venv.html
[Install TensorFlow]: https://www.tensorflow.org/install/
