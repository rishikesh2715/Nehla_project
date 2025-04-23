# Case Approval Time Prediction

This project aims to predict the number of days it will take for a case to get approved using historical review data. It was developed as part of a machine learning pipeline to support a PhD research workflow on investigation timelines.

---

## Project Overview

- ✅ Filters only **approved cases** (e.g., "Approved for Mailout", etc.)
- ✅ Computes **Processing Time (Days)** between submission and approval
- ✅ Builds a **regression model** (RandomForest, HistGB, etc.)
- ✅ Evaluates with **MAE** and **R²**
- ✅ Visualizes case timelines, distributions, and review breakdowns
- ✅ Predicts duration for **new unseen cases**
- ✅ Includes **model saving/loading with pickle**

---

## Folder Structure
```bash
├── data/ 
  ├── updated_Dataset.xlsx # Historical data with labeled statuses 
│ └── new_cases.xlsx # New incoming cases to predict on 
├── src/ 
│ ├── data_loader.py # Loads and preprocesses data 
│ ├── feature_engineering.py # Creates time-based & encoded features 
│ ├── model.py # Modeling, evaluation, saving/loading 
│ └── visualizer.py # EDA & presentation-ready visual plots 
├── main.py # Full training + EDA + saving pipeline 
├── predict.py # Inference script to predict on new data 
├── saved_model.pkl # Serialized trained model 
└── README.md # This file

```


---

## How to Run

### 1. Train Model
```bash
python main.py
```
###  2. Predict on New Cases
```bash
python predict.py
```
### 3. View Results
* Check the `saved_model.pkl` for the trained model.

* The output plots will be stored in the `results/` directory.

## Format to Predict on New Cases

Drop a new Excel file in the `data/` folder with columns like:

```plaintext
Case Number | Date Received | Section | Assigned EC | Expedited Review | Type of Review | QA Reviewer Assigned | Deputy Director Review

```

## Requirements
```bash
pip install -r requirements.txt
```




