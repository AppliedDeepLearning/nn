# Losses

Loss functions are available under `nn.losses`.


## Usage

To use a loss function for a model, you can pass its name as an argument:

```py
nn.Model(network, loss='softmax_cross_entropy', ...)
```

Or pass a loss function:

```py
nn.Model(network, loss=nn.losses.softmax_cross_entropy, ...)
```

You can also use a custom function:

```py
def custom_loss(labels, outputs):
    # Compute loss
    return loss

nn.Model(network, loss=custom_loss, ...)
```


## Available Losses

- `absolute_difference`
- `add_loss`
- `compute_weighted_loss`
- `cosine_distance`
- `hinge_loss`
- `huber_loss`
- `log_loss`
- `mean_pairwise_squared_error`
- `mean_squared_error`
- `sigmoid_cross_entropy`
- `softmax_cross_entropy`
- `sparse_softmax_cross_entropy`


## See Also

- [TensorFlow Losses](https://www.tensorflow.org/api_docs/python/tf/losses#functions){:target="_blank"}
