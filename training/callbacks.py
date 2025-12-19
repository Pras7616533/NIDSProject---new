from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def get_callbacks():
    """
    Returns callbacks for training
    """

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath='saved_models/dnn_best_model.h5',
        monitor='val_loss',
        save_best_only=True
    )

    return [early_stop, checkpoint]
