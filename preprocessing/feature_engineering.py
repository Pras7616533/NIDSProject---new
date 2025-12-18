from sklearn.preprocessing import LabelEncoder
import numpy as np


def encode_features(df):
    print("\nStarting feature engineering...")

    df = df.copy()

    # Encode categorical columns (by index)
    categorical_columns = [1, 2, 3]
    for col_index in categorical_columns:
        le = LabelEncoder()
        df.iloc[:, col_index] = le.fit_transform(df.iloc[:, col_index])

    # ---- BINARY LABEL CONVERSION ----
    # normal -> 0, attack -> 1
    df.iloc[:, -1] = df.iloc[:, -1].apply(
        lambda x: 0 if x == "normal" else 1
    )

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print("Binary classification enabled")
    print(f"Classes: {np.unique(y)}")

    return X, y
