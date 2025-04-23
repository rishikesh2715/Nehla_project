# src/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_processing_time_by_review_type(df):
    """
    Plot processing time by type of review using a strip plot.
    This function visualizes the distribution of processing times
    for different types of reviews.

    :param
        df: DataFrame containing the data to plot.
        :type df: pd.DataFrame
        :return: None
    
    """
    plt.figure(figsize=(10, 6))
    sns.stripplot(
        data=df,
        x='Type of Review',
        y='Processing Time (Days)',
        jitter=True,
        alpha=0.6
    )
    plt.title('Processing Time vs Type of Review')
    plt.xlabel('Type of Review')
    plt.ylabel('Processing Time (Days)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/processing_time_by_review_type.png', dpi=300)
    plt.show()

def plot_boxplot_by_section(df):
    """
    Plot processing time by section using a box plot.
    This function visualizes the distribution of processing times
    for different sections.

    :param
        df: DataFrame containing the data to plot.
        :type df: pd.DataFrame
        :return: None

    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Processing Time (Days)', y='Section')
    plt.title('Processing Time by Section')
    plt.xlabel('Processing Time (Days)')
    plt.ylabel('Section')
    plt.tight_layout()
    plt.savefig('results/processing_time_by_section.png', dpi=300)
    plt.show()

def plot_heatmap_weekday_month(df):
    """
    Plot a heatmap showing the number of submissions by weekday and month.
    This function visualizes the distribution of submissions
    across different weekdays and months.

    :param
        df: DataFrame containing the data to plot.
        :type df: pd.DataFrame
        :return: None

    """
    df = df.copy()
    df['Weekday'] = df['Submission Date'].dt.day_name()
    df['Month'] = df['Submission Date'].dt.month_name()

    pivot = df.pivot_table(
        index='Weekday',
        columns='Month',
        values='Case Number',  # Case Number is a safe count anchor
        aggfunc='count'
    )

    # Reorder weekdays
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(weekdays)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot.fillna(0).astype(int), annot=True, fmt='d', cmap='Blues')
    plt.title('Submissions Heatmap: Weekday vs Month')
    plt.xlabel('Month')
    plt.ylabel('Weekday')
    plt.tight_layout()
    plt.savefig('results/submissions_heatmap_weekday_month.png', dpi=300)
    plt.show()


def plot_residuals(model, X_test, y_test):
    """
    Plot residuals of the model predictions.
    This function visualizes the difference between actual and predicted
    processing times.

    :param
        model: Trained model to evaluate.
        :type model: sklearn estimator
        :param X_test: Test features.
        :type X_test: pd.DataFrame
        :param y_test: Actual target values.
        :type y_test: pd.Series
        :return: None

    """
    # Residual scatter plot
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted Processing Time (Days)')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig('results/residuals_vs_predicted.png', dpi=300)
    plt.show()


def plot_boxplot_by_flag(df, column_name, title=None):
    """
    Plot processing time by a specific flag (e.g., Expedited Review).
    This function visualizes the distribution of processing times
    based on a binary flag.
    
    :param
        df: DataFrame containing the data to plot.
        :type df: pd.DataFrame
        :param column_name: Name of the column to use as a flag.
        :type column_name: str
        :param title: Title of the plot.
        :type title: str
        :return: None

    """
    if column_name not in df.columns:
        return
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x=f'{column_name}_Flag',
        y='Processing Time (Days)'
    )
    plt.title(title or f'Processing Time by {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Processing Time (Days)')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.tight_layout()
    plt.savefig(f'results/processing_time_by_{column_name}.png', dpi=300)
    plt.show()

def plot_violin_by_review_type(df):
    """
    Plot processing time by type of review using a violin plot.
    This function visualizes the distribution of processing times
    for different types of reviews.
    :param
        df: DataFrame containing the data to plot.
        :type df: pd.DataFrame
        :return: None

    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x='Type of Review',
        y='Processing Time (Days)',
        inner='quartile'
    )
    plt.title('Violin Plot: Processing Time by Review Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/violin_plot_processing_time_by_review_type.png', dpi=300)
    plt.show()

def plot_monthly_submission_trend(df):
    """
    Plot the trend of submissions over months.
    This function visualizes the number of submissions
    for each month in the dataset.
    :param
        df: DataFrame containing the data to plot.
        :type df: pd.DataFrame
        :return: None

    """
    df = df.copy()
    df['Month'] = df['Submission Date'].dt.to_period('M')
    trend = df.groupby('Month').size()
    plt.figure(figsize=(10, 5))
    trend.plot(kind='line', marker='o')
    plt.title('Monthly Submission Trend')
    plt.xlabel('Month')
    plt.ylabel('Number of Cases')
    plt.tight_layout()
    plt.show()

def plot_case_count_by_column(df, column_name, title=None):
    """
    Plot the count of cases by a specific column (e.g., Assigned EC).
    This function visualizes the number of cases for each unique value
    in the specified column.

    :param
        df: DataFrame containing the data to plot.
        :type df: pd.DataFrame
        :param column_name: Name of the column to use for counting.
        :type column_name: str
        :param title: Title of the plot.
        :type title: str
    
    """
    if column_name not in df.columns:
        return
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y=column_name, order=df[column_name].value_counts().index)
    plt.title(title or f'Case Count by {column_name}')
    plt.xlabel('Count')
    plt.ylabel(column_name)
    plt.tight_layout()
    plt.savefig(f'results/case_count_by_{column_name}.png', dpi=300)
    plt.show()

def plot_processing_time_hist(df):
    """
    Plot a histogram of processing times.
    This function visualizes the distribution of processing times
    in the dataset.

    :param
        df: DataFrame containing the data to plot.
        :type df: pd.DataFrame
        :return: None

    """
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Processing Time (Days)'], bins=30, kde=True)
    plt.title('Distribution of Processing Time (Days)')
    plt.xlabel('Processing Time (Days)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('results/processing_time_histogram.png', dpi=300)
    plt.show()

def plot_processing_time_by_coordinator(df):
    """
    Plot processing time by investigator (Assigned EC).
    This function visualizes the distribution of processing times
    for different investigators.
    :param
        df: DataFrame containing the data to plot.
        :type df: pd.DataFrame
        :return: None

    """
    if 'Assigned EC' not in df.columns:
        return
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Processing Time (Days)', y='Assigned EC')
    plt.title('Processing Time by Investigator (Assigned EC)')
    plt.xlabel('Processing Time (Days)')
    plt.ylabel('Assigned EC')
    plt.tight_layout()
    plt.savefig('results/processing_time_by_investigator.png', dpi=300)
    plt.show()


