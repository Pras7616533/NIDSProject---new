from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

def load_nids_model(weights_path):
    model = Sequential([
        Input(shape=(41,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(23, activation='softmax')
    ])

    model.load_weights(weights_path)
    return model
