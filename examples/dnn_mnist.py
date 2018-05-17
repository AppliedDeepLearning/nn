import nn

# Allow hyper parameters and other options to be set from command-line
args = nn.cli(learning_rate=0.001,
              epochs=30,
              batch_size=128,
              directory=None)

# Prepare data
(x_train, y_train), (x_test, y_test) = nn.datasets.mnist.load_data()
x_train = nn.np.asarray(x_train, dtype='float')
x_test = nn.np.asarray(x_test, dtype='float')
y_train = nn.np.asarray(y_train, dtype='int')
y_test = nn.np.asarray(y_test, dtype='int')


# Create the model
@nn.model(directory=args.directory, params=vars(args))
def model(x, params):
    # Define the network architecture
    x = nn.Flatten()(x)
    x = nn.Dense(units=64, activation='relu')(x)
    x = nn.Dense(units=32, activation='relu')(x)
    outputs = nn.Dense(units=10)(x)
    # Compute predictions for prediction mode
    predictions = nn.tf.argmax(outputs, axis=1)

    # Configure the learning process
    return dict(outputs=outputs,
                predictions=predictions,
                loss='sparse_softmax_cross_entropy',
                optimizer=('GradientDescent', params['learning_rate']),
                metrics=['accuracy'])

# Train the model using training data
model.train(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

# Evaluate the model performance on test or validation data
print(model.evaluate(x_test, y_test))

# Use the model to make predictions
print(list(model(x_test[0:10])))
