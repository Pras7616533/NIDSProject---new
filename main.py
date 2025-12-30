import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from preprocessing.data_cleaning import clean_data
from preprocessing.feature_engineering import encode_features
from models.dnn_model import build_dnn
from training.callbacks import get_callbacks
from evaluation.evaluate_model import evaluate_model

def main():
    print("\nLoading dataset...")
    df = pd.read_csv("data/raw/NSL_KDD.csv")
    print(f"Dataset loaded successfully")
    print(f"Shape: {df.shape}")

    # ---------------- STEP 4: Data Cleaning ----------------
    df = clean_data(df)

    # ---------------- STEP 6: Feature Engineering ----------------
    X, y = encode_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")

    # ---------------- STEP 7: Feature Scaling ----------------
    print("\nStarting feature scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Feature scaling completed")
    print(f"Scaled feature shape: {X_scaled.shape}")

    # ---------------- STEP 7.5: Compute Class Weights ----------------
    print("\nComputing class weights...")
    classes = np.unique(y)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights)}
    print("Class weights computed:")
    print(class_weights)

    # ---------------- STEP 7.6: Train / Val / Test Split ----------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("\nDataset split completed:")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Testing set: {X_test.shape}")

    # ---------------- IMPORTANT FIX: DATA TYPE CONVERSION ----------------
    X_train = X_train.astype(np.float32)
    X_val   = X_val.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    y_train = np.array(y_train).astype(np.float32)
    y_val   = np.array(y_val).astype(np.float32)
    y_test  = np.array(y_test).astype(np.float32)

    # ---------------- STEP 8: Build DNN Model ----------------
    print("\nBuilding DNN model...")
    input_dim = X_train.shape[1]
    model = build_dnn(input_dim)

    print("\nDNN Model Summary:")
    model.summary()

    # ---------------- STEP 9: Train the Model ----------------
    print("\nStarting model training...")
    callbacks = get_callbacks()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=128,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("\nModel training completed successfully!")

    # ---------------- SAVE FINAL MODEL ----------------
    model.save("saved_models/original_model.h5")
    print("Final model saved as dnn_final_model.h5")

    # ---------------- STEP 10: Model Evaluation ----------------
    evaluate_model(model, X_test, y_test)



if __name__ == "__main__":
    main()
