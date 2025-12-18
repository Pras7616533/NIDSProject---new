import pandas as pd


def load_dataset(file_path):
    """
    Load network intrusion dataset from CSV file

    Parameters:
    file_path (str): Path to dataset CSV

    Returns:
    df (DataFrame): Loaded dataset
    """

    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully")
        print(f"Shape: {df.shape}")
        return df

    except Exception as e:
        print("Error loading dataset:", e)
        return None
