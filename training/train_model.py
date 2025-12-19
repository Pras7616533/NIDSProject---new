def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    """
    Train the DNN model
    """

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=128,
        class_weight=class_weights,
        callbacks=None,
        verbose=1
    )

    return history
