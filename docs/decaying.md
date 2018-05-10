# Decaying Learning Rate

Decay functions can be applied to learning rate using `nn.decay` and `nn.optimizer` utility functions.

```py
nn.decay(fn_name, **kwargs)
```

`**kwargs` are passed to the decay function.


## Usage

```py
decay = nn.decay('exponential', decay_steps=1000, decay_rate=0.96)

optimizer = nn.optimizer('GradientDescent', 0.001, decay=decay)

nn.Model(network, optimizer=optimizer, ...)
```

You can also use a custom function:

```py
def custom_decay(learning_rate, global_step):
    # Custom logic
    return decayed_learning_rate

optimizer = nn.optimizer('GradientDescent', 0.001, decay=custom_decay)

nn.Model(network, optimizer=optimizer, ...)
```


## Available Decay Functions

- `exponential`
- `inverse_time`
- `natural_exp`
- `polynomial`
- `cosine`
- `linear_cosine`
- `noisy_linear_cosine`


## See Also

- [TensorFlow Decay Functions](https://www.tensorflow.org/api_guides/python/train#Decaying_the_learning_rate){:target="_blank"}
