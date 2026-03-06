from tensorflow.keras import Sequential, layers,regularizers

def build_model():
    model = Sequential([
        layers.Conv1D(32, 5, activation='relu', input_shape=(960,1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.007)),
        layers.Dropout(0.4),
        layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])
    return model