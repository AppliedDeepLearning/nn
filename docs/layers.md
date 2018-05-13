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


## Recurrent Layers

- `RNN`
- `BasicLSTMCell`
- `BasicRNNCell`
- `DeviceWrapper`
- `DropoutWrapper`
- `GRUCell`
- `LSTMCell`
- `LSTMStateTuple`
- `MultiRNNCell`
- `RNNCell`
- `ResidualWrapper`

### RNN

```py
nn.RNN(cell, return_sequences=False, return_state=False, **kwargs)
```

### Example

```py
def model(x):
    # Create layers
    embedding = nn.Embedding(10000, 300)
    cell = nn.LSTMCell(128)
    rnn = nn.RNN(cell)
    # Connect layers
    sequence_length = nn.sequence.length(x)  # required for variable length sequences
    x = embedding(x)
    x = rnn(x, sequence_length=sequence_length)
    ...
```

## Sparse Layers

- `Embedding`

### Embedding

```py
nn.Embedding(input_dim, output_dim, embeddings_initializer=None, embeddings_regularizer=None, embeddings_constraint=None, dtype=tf.float32, **kwargs)
```


## Normalization Layers

- `BatchNormalization`


## See Also

- [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers#classes){:target="_blank"}
- [TensorFlow RNN Cells](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell#classes){:target="_blank"}
