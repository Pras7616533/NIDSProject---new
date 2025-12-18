from preprocessing.data_loader import load_dataset
from preprocessing.data_cleaning import clean_data
from preprocessing.feature_engineering import encode_features
from preprocessing.scaler import scale_features
from preprocessing.imbalance_handler import compute_class_weights
from models.dnn_model import build_dnn
from sklearn.model_selection import train_test_split


def main():
    dataset_path = "data/raw/NSL_KDD.csv"

    df = load_dataset(dataset_path)

    if df is not None:
        df = clean_data(df)
        X, y = encode_features(df)
        X_scaled, scaler = scale_features(X)
        class_weights = compute_class_weights(y)

        # Step 7: Train / Validation / Test split
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
        
        # Step 8: Build DNN model
        input_dim = X_train.shape[1]
        model = build_dnn(input_dim)

        print("\nDNN Model Summary:")
        model.summary()



if __name__ == "__main__":
    main()
