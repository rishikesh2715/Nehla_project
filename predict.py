# predict.py

import pandas as pd
from src.data_loader import preprocess_data
from src.feature_engineering import create_features
from src.model import load_model

# Load new unseen data
new_data_path = "data/new_data.xlsx"  # Change this to your file
df = pd.read_excel(new_data_path)
df.columns = df.columns.str.strip()

# Preprocess & feature-engineer the new data
df = preprocess_data(df)
df = create_features(df)

# Load saved model
model = load_model("saved_model.pkl")

# Define features used in training
feature_cols = [
    c for c in df.columns
    if c.startswith("Submission_") or
       c.endswith("_Freq") or
       c.endswith("_Flag")
]

# Make sure features exist
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

# Predict
preds = model.predict(df[feature_cols])
df["Predicted Processing Time (Days)"] = preds

# Show result
print(df[["Case Number", "Predicted Processing Time (Days)"]].head())
