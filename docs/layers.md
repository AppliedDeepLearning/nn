# Layers

Layers are the building blocks of a neural network. Layer classes are available under `nn.layers` and also aliased under `nn`.


## Usage

To use a layer, create a layer instance and call it directly:

```py
layer = nn.Dense(units=64, activation='relu')
outputs = layer(inputs)
```

Or in a single line:

```py
outputs = nn.Dense(units=64, activation='relu')(inputs)
```


## Core Layers

- `Dense`
- `Dropout`
- `Flatten`
- `InputSpec`


## Convolutional Layers

- `Conv1D`
- `Conv2D`
- `Conv2DTranspose`
- `Conv3D`
- `Conv3DTranspose`
- `SeparableConv1D`
- `SeparableConv2D`


## Pooling Layers

- `AveragePooling1D`
- `AveragePooling2D`
- `AveragePooling3D`
- `MaxPooling1D`
- `MaxPooling2D`
- `MaxPooling3D`


## Normalization Layers

- `BatchNormalization`


## See Also

- [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers#classes){:target="_blank"}
