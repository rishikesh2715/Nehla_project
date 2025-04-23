# src/data_loader.py

import pandas as pd

def load_data(filepath):
    """
    Load data from an Excel file and clean column names.

    :param filepath: Path to the Excel file.
    :type filepath: str
    :return: DataFrame with cleaned column names.
    :rtype: pd.DataFrame
    """
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    return df

def preprocess_data(df):
    """
    Preprocess the dataset by:
    1. Renaming columns for consistency.
    2. Converting date columns to datetime format.
    3. Filtering to only include approved cases.
    4. Dropping rows with missing dates.
    5. Calculating processing time in days.
    6. Dropping rows with negative or null processing time.

    :param df: DataFrame containing the dataset.
    :type df: pd.DataFrame
    :return: Preprocessed DataFrame.
    :rtype: pd.DataFrame
    """
    df = df.copy()

    # 1. Rename for consistency
    df.rename(columns={
        "Date Received": "Submission Date",
        "Modified": "Last Modified"
    }, inplace=True)

    # 2. Convert to datetime
    df["Submission Date"] = pd.to_datetime(df["Submission Date"], errors="coerce")
    df["Last Modified"] = pd.to_datetime(df["Last Modified"], errors="coerce")

    # 3. Define which statuses count as "approved"
    approved_statuses = [
        "Approved for Mailout",
        "Approved for Mailout with Changes",
        "DD - Approved for Mailout",
        "DD - Approved for Mailout with Changes",
    ]

    # 4. Filter to only approved cases
    df = df[df["Status"].isin(approved_statuses)]

    # 5. Drop rows with missing dates
    df.dropna(subset=["Submission Date", "Last Modified"], inplace=True)

    # 6. Calculate processing time
    df["Processing Time (Days)"] = (
        df["Last Modified"] - df["Submission Date"]
    ).dt.days

    # 7. Drop rows with negative or null processing time
    df = df[df["Processing Time (Days)"] >= 0]

    return df
