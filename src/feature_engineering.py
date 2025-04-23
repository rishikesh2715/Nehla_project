# src/feature_engineering.py

def create_features(df):
    """
    Create additional features for the dataset.
    - Time-based features from Submission Date
    - Frequency encoding for key roles
    - Binary encoding for Yes/No columns

    :param df: DataFrame containing the dataset.
    :type df: pd.DataFrame
    :return: DataFrame with additional features.

    """
    df = df.copy()

    # 1. Time-based features from Submission Date
    df['Submission_Weekday'] = df['Submission Date'].dt.dayofweek
    df['Submission_Month'] = df['Submission Date'].dt.month
    df['Submission_Quarter'] = df['Submission Date'].dt.quarter
    df['Submission_Year'] = df['Submission Date'].dt.year

    # 2. Frequency encoding for key roles
    for col in ['Section', 'Assigned EC', 'QA Reviewer Assigned']:
        if col in df.columns:
            freq = df[col].value_counts()
            df[f'{col}_Freq'] = df[col].map(freq)

    # 3. Binary encoding for Yes/No columns
    for col in ['Expedited Review', 'Deputy Director Review']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[f'{col}_Flag'] = df[col].map({'yes': 1, 'no': 0})

    return df
