A neural network library built on top of TensorFlow for quickly building deep learning models.


## Installation

```sh
pip install nn
```


## Example

```py
import nn

# Create the model
@nn.model
def model(inputs):
    # Define the network architecture (layers, number of units, activations)
    hidden = nn.Dense(units=64, activation='relu')(inputs)
    outputs = nn.Dense(units=10)(hidden)

    # Configure the learning process (loss, optimizer, evaluation metrics)
    return dict(outputs=outputs,
                loss='softmax_cross_entropy',
                optimizer=('GradientDescent', 0.001),
                metrics=['accuracy'])

# Train the model using training data:
model.train(x_train, y_train, epochs=30, batch_size=128)

# Evaluate the model performance on test or validation data:
loss_and_metrics = model.evaluate(x_test, y_test)

# Use the model to make predictions for new data:
predictions = model.predict(x)
```


## Documentation

See [documentation].


## License

[MIT][license]


[license]: /LICENSE
[documentation]: https://nn.applieddeeplearning.com/
