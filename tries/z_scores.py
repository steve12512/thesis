import pandas as pd
import numpy as np
from scipy import stats

# Function to calculate Z-scores for each column in the DataFrame
def calculate_z_scores(df):
    # Ensure all columns are numeric
    df_numeric = df.select_dtypes(include=[np.number])
    
    def z_score(column):
        mean = np.mean(column)
        std_dev = np.std(column)
        z_scores = (column - mean) / std_dev
        return z_scores
    
    z_scores_results = {}
    for col in df_numeric.columns:
        z_scores_results[col] = z_score(df_numeric[col])
    
    return pd.DataFrame(z_scores_results)

# Function to check if the Z-scores follow a normal distribution
def check_normality(z_scores_df):
    def test_normality(z_scores):
        stat, p_value = stats.shapiro(z_scores)
        return p_value > 0.05

    normality_results = {}
    for column in z_scores_df.columns:
        normality_results[column] = test_normality(z_scores_df[column])
    
    return normality_results

# Main function to calculate Z-scores and check normality
def analyze_columns(df):
    z_scores_df = calculate_z_scores(df)
    normality_results = check_normality(z_scores_df)
    
    return normality_results
