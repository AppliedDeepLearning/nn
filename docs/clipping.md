# Gradient Clipping

Clipping functions can be applied to gradients (for handling exploding gradients) using `nn.clip` and `nn.optimizer` utility functions.

```py
nn.clip(fn_name, **kwargs)
```

`**kwargs` are passed to the clipping function.


## Usage

```py
clip = nn.clip('value', clip_value_min=-5, clip_value_max=5)

optimizer = nn.optimizer('GradientDescent', 0.001, clip=clip)
```

You can also use a custom function:

```py
def custom_clip(values):
    # Custom logic
    return clipped_values

optimizer = nn.optimizer('GradientDescent', 0.001, clip=custom_clip)
```

> **Note:** Order of values should remain same after clipping.


## Available Clipping Functions

- `value`
- `norm`
- `average_norm`
- `global_norm`


## See Also

- [TensorFlow Clipping Functions](https://www.tensorflow.org/api_guides/python/train#Gradient_Clipping){:target="_blank"}
