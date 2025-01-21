import pandas as pd
import numpy as np

def gini_index(series):
    """Calculate the Gini index for a categorical series."""
    prop = series.value_counts(normalize=True)  # Proportion of each category
    gini = 1 - np.sum(prop ** 2)                # Gini index formula
    return gini

def calculate_gini_indices(df):
    # List of categorical columns
    categorical_columns = ['Gender', 'SMOKE', 'family_history', 'FAVC', 'CAEC', 'SCC', 'CALC', 'MTRANS', 'Obesity']
    
    # Create an empty DataFrame to store Gini index results
    gini_results = pd.DataFrame(columns=['Feature', 'Gini Index'])

    # Calculate Gini index for each categorical column
    for col in categorical_columns:
        gini_value = gini_index(df[col])
        # Using pd.concat to append rows
        gini_results = pd.concat([gini_results, pd.DataFrame({'Feature': col, 'Gini Index': gini_value}, index=[0])], ignore_index=True)

    return gini_results
