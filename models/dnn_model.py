from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_dnn(input_dim):
    """
    Build Deep Neural Network for Binary Intrusion Detection

    Parameters:
    input_dim (int): Number of input features

    Returns:
    model (Sequential): Compiled DNN model
    """

    model = Sequential(name="DeepNIDS_DNN")

    # Input + Hidden layers
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    # Output layer (Binary classification)
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
