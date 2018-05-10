# Activations

Activation functions are available under `nn.activations`.


## Usage

To use an activation function for a layer, you can pass its name as an argument:

```py
nn.Dense(units=64, activation='relu')
```

Or pass an activation function:

```py
nn.Dense(units=64, activation=nn.activations.relu)
```


## Available Activations

- `relu`
- `relu6`
- `crelu`
- `elu`
- `selu`
- `softplus`
- `softsign`
- `softmax`
- `sigmoid`
- `tanh`


## See Also

- [TensorFlow Activations](https://www.tensorflow.org/api_guides/python/nn#Activation_Functions){:target="_blank"}
