from sklearn.preprocessing import StandardScaler
import numpy as np


def scale_features(X):
    """
    Apply StandardScaler to feature matrix

    Parameters:
    X (DataFrame): Feature matrix

    Returns:
    X_scaled (ndarray): Scaled features
    scaler (StandardScaler): Fitted scaler object
    """

    print("\nStarting feature scaling...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Feature scaling completed")
    print(f"Scaled feature shape: {X_scaled.shape}")

    return X_scaled, scaler
