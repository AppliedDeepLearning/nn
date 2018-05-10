# Metrics

Metric functions are available under `nn.metrics`.


## Usage

`nn.Model` expects a `list` or `dict` of metrics. To use a metric function for a model, you can pass its name as an argument:

```py
nn.Model(network, metrics=['accuracy'], ...)
```

Or pass a metric function:

```py
nn.Model(network, metrics=[nn.metrics.accuracy], ...)
```

You can also use a custom function:

```py
def custom_metric(labels, outputs):
    # Compute metric
    return metric

nn.Model(network, metrics=['accuracy', custom_metric], ...)
```

To give a custom name for each metric, pass a `dict` of metrics with keys as metric names and values as metric functions:

```py
nn.Model(network, metrics={'my_metric': custom_metric}, ...)
```


## Available Metrics

- `accuracy`
- `auc`
- `average_precision_at_k`
- `false_negatives`
- `false_negatives_at_thresholds`
- `false_positives`
- `false_positives_at_thresholds`
- `mean`
- `mean_absolute_error`
- `mean_cosine_distance`
- `mean_iou`
- `mean_per_class_accuracy`
- `mean_relative_error`
- `mean_squared_error`
- `mean_tensor`
- `percentage_below`
- `precision`
- `precision_at_k`
- `precision_at_thresholds`
- `precision_at_top_k`
- `recall`
- `recall_at_k`
- `recall_at_thresholds`
- `recall_at_top_k`
- `root_mean_squared_error`
- `sensitivity_at_specificity`
- `sparse_average_precision_at_k`
- `sparse_precision_at_k`
- `specificity_at_sensitivity`
- `true_negatives`
- `true_negatives_at_thresholds`
- `true_positives`
- `true_positives_at_thresholds`


## See Also

- [TensorFlow Metrics](https://www.tensorflow.org/api_docs/python/tf/metrics#functions){:target="_blank"}
