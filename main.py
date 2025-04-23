from src.data_loader import load_data, preprocess_data
from src.feature_engineering import create_features
from src.model import (
    clean_and_transform_target,
    evaluate_models,
    get_regressors,
    save_model
)

from src.visualizer import (
    plot_processing_time_by_review_type,
    plot_boxplot_by_section,
    plot_heatmap_weekday_month,
    plot_boxplot_by_flag,
    plot_violin_by_review_type,
    plot_residuals,
    plot_monthly_submission_trend,
    plot_case_count_by_column,
    plot_processing_time_hist,
    plot_processing_time_by_coordinator
)

from sklearn.model_selection import train_test_split

# Updated dataset path
FILEPATH = 'data/updated_Dataset.xlsx'

def main():
    """
    Main function to run the entire pipeline:
    1. Load and preprocess data
    2. Perform exploratory data analysis (EDA)
    3. Clean and transform target variable
    4. Select features
    5. Evaluate models
    6. Train the best model and plot residuals
    """
    # 1. Load and preprocess
    df = load_data(FILEPATH)
    df = preprocess_data(df)
    df = create_features(df)

    # 2. Drop unused columns that might still be floating around
    df = df.dropna(subset=['Processing Time (Days)'])

    
    # Run EDA plots
    plot_processing_time_by_review_type(df)
    plot_boxplot_by_section(df)
    plot_heatmap_weekday_month(df)
    plot_boxplot_by_flag(df, 'Expedited Review', 'Processing Time by Expedited Review')
    plot_boxplot_by_flag(df, 'Deputy Director Review', 'Processing Time by Deputy Director Review')
    plot_violin_by_review_type(df)
    plot_monthly_submission_trend(df)
    plot_case_count_by_column(df, 'Assigned EC', 'Case Count by Investigator (Assigned EC)')
    plot_case_count_by_column(df, 'Section', 'Case Count by Section')
    plot_processing_time_hist(df)
    plot_processing_time_by_coordinator(df)


    # 3. Clean target
    df_clean, target = clean_and_transform_target(df, log_transform=False)

    # 4. Select features
    feature_cols = [
        c for c in df_clean.columns
        if c.startswith('Submission_') or
           c.endswith('_Freq') or
           c.endswith('_Flag')
    ]
    df_clean[feature_cols] = df_clean[feature_cols].fillna(df_clean[feature_cols].median())

    # 5. Evaluate models
    results = evaluate_models(df_clean, feature_cols, target)
    print("\n=== Model Comparison ===")
    print(results.to_string(index=False))

    # 6. Train best model and plot residuals
    best_name = results.iloc[0]['Model']
    best_model = get_regressors()[best_name]

    X = df_clean[feature_cols]
    y = df_clean[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)

    save_model(best_model, "saved_model.pkl")

    plot_residuals(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
