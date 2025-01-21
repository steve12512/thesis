
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt

def calculate_entropy(column, bins=None):
    """
    Calculate the entropy of a column.
    - For categorical variables: normalized entropy.
    - For continuous variables: entropy based on binning.
    """
    if column.dtype in [np.float64, np.int64] or np.issubdtype(column.dtype, np.number):  # Continuous variables
        counts, _ = np.histogram(column.dropna(), bins=bins or 10)
    else:  # Categorical variables
        counts = column.value_counts().values
    
    counts = counts[counts > 0]  # Remove zero counts to avoid log(0)
    prob_dist = counts / counts.sum()
    return entropy(prob_dist, base=2)

def analyze_dataset_entropy(df, bins=10):
    """
    Analyze the entropy of each column in the dataset.
    Returns a DataFrame with columns and their entropies.
    """
    results = []

    for col in df.columns:
        try:
            col_entropy = calculate_entropy(df[col], bins=bins)
            results.append({
                'Column': col,
                'Entropy': col_entropy
            })
        except Exception as e:
            results.append({
                'Column': col,
                'Entropy': None,
                'Error': str(e)
            })

    result_df = pd.DataFrame(results)
    return result_df

def plot_column_entropy(entropy_df):
    """
    Plot entropy values for each column.

    Parameters:
        entropy_df (pd.DataFrame): DataFrame containing column entropy values.
    """
    entropy_df.sort_values(by='Entropy', ascending=False, inplace=True)
    plt.figure(figsize=(10, 6))
    plt.bar(entropy_df['Column'], entropy_df['Entropy'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Entropy of Dataset Columns')
    plt.ylabel('Normalized Entropy')
    plt.xlabel('Columns')
    plt.tight_layout()
    plt.show()

# Example usage:
# Load your dataset
# df = pd.read_csv("your_dataset.csv")
# entropy_df = analyze_dataset_entropy(df)
# plot_column_entropy(entropy_df)
# print(entropy_df)
