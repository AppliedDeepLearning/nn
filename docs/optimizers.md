# Optimizers

Optimizer classes are available under `nn.optimizers`.


## Usage

To use an optimizer for a model, you can pass a tuple of optimizer name and learning rate as an argument:

```py
return dict(optimizer=('GradientDescent', 0.001), ...)
```

Or pass an optimizer instance:

```py
return dict(optimizer=nn.optimizers.GradientDescent(0.001), ...)
```

It is recommended to use the `nn.optimizer` utility function as it allows to specify more operations like [Decaying Learning Rate] and [Gradient Clipping]:

```py
return dict(optimizer=nn.optimizer('GradientDescent', 0.001), ...)
```

It has the following signature:

```py
nn.optimizer(class_name, learning_rate, **kwargs)
```

`**kwargs` are passed to the optimizer class.

You can also use a custom function that returns a training operation:

```py
def custom_optimizer(loss, global_step):
    optimizer = nn.optimizers.GradientDescent(0.001)
    train_op = optimizer.minimize(loss=loss, global_step=global_step)
    return train_op

# Inside model
return dict(optimizer=custom_optimizer, ...)
```

Inside this custom function, you can specify other operations like decaying learning rate and gradient clipping.


## Available Optimizers

- `GradientDescent`
- `Adadelta`
- `Adagrad`
- `AdagradDA`
- `Momentum`
- `Adam`
- `Ftrl`
- `ProximalGradientDescent`
- `ProximalAdagrad`
- `RMSProp`


## Next Steps

- [Learn more about Decaying Learning Rate][Decaying Learning Rate]
- [Learn more about Gradient Clipping][Gradient Clipping]


## See Also

- [TensorFlow Optimizers](https://www.tensorflow.org/api_guides/python/train#Optimizers){:target="_blank"}


[Decaying Learning Rate]: ../decaying/
[Gradient Clipping]: ../clipping
