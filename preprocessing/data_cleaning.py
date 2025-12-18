import pandas as pd


def clean_data(df):
    """
    Perform basic data cleaning:
    - Handle missing values
    - Remove duplicate records

    Parameters:
    df (DataFrame): Raw dataset

    Returns:
    df (DataFrame): Cleaned dataset
    """

    print("\nStarting data cleaning process...")

    # 1. Check missing values
    missing_values = df.isnull().sum().sum()
    print(f"Total missing values before cleaning: {missing_values}")

    # If missing values exist, fill with 0 (safe for numeric network data)
    if missing_values > 0:
        df = df.fillna(0)
        print("Missing values filled with 0")

    # 2. Remove duplicate rows
    duplicate_rows = df.duplicated().sum()
    print(f"Duplicate rows before cleaning: {duplicate_rows}")

    if duplicate_rows > 0:
        df = df.drop_duplicates()
        print("Duplicate rows removed")

    print("Data cleaning completed")
    print(f"Dataset shape after cleaning: {df.shape}")

    return df
