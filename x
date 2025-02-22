import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference, demographic_parity_ratio
from sklearn.impute import SimpleImputer
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import BinaryLabelDatasetMetric

# Load data
df = pd.read_csv('car_insurance_claim.csv')

# Clean data
df = df.drop(columns=['ID', 'BIRTH'], axis=1)

# Define numerical and categorical columns
numerical = [
    'KIDSDRIV', 'AGE', 'HOMEKIDS', 'YOJ', 'INCOME',
    'HOME_VAL', 'TRAVTIME', 'BLUEBOOK', 'TIF', 'OLDCLAIM',
    'CLM_FREQ', 'MVR_PTS', 'CLM_AMT', 'CAR_AGE'
]

categorical = [
    'PARENT1', 'MSTATUS', 'GENDER', 'EDUCATION',
    'OCCUPATION', 'CAR_USE', 'CAR_TYPE', 'RED_CAR', 'REVOKED', 'URBANICITY'
]

# Handle missing values in categorical columns
df[categorical] = df[categorical].apply(lambda x: x.fillna(x.mode()[0]))

# Clean currency columns
def clean_currency(x):
    if isinstance(x, str):
        return float(x.replace('$', '').replace(',', ''))
    return x

for col in ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']:
    df[col] = df[col].apply(clean_currency)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=categorical, drop_first=False)

# Define new categorical columns after one-hot encoding
new_categorical = [col for col in df.columns if col not in numerical and col != 'CLAIM_FLAG']

# Ensure all new categorical columns are numeric
df[new_categorical] = df[new_categorical].astype(int)

# Split data into features and target
X = df.drop(columns=['CLAIM_FLAG'], axis=1)
y = df['CLAIM_FLAG']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical),
        ("cat", Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ]), new_categorical)
    ]
)

# Model pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Evaluate model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Fairness evaluation
groups = ['GENDER', 'EDUCATION', 'MSTATUS', 'PARENT1', 'OCCUPATION', 'URBANICITY']
for group in groups:
    evaluate_fairness(y_test, y_pred, X_test[group], group)

# Bias mitigation using Reweighing
# Convert dataset into AIF360 format
dataset = StandardDataset(
    df,
    label_name='CLAIM_FLAG',
    protected_attribute_names=['GENDER', 'EDUCATION', 'PARENT1'],
    favorable_classes=[1],
    privileged_classes=[{'GENDER': [1]}, {'EDUCATION': [1]}, {'PARENT1': [1]}]
)

# Apply Reweighing
rw = Reweighing(unprivileged_groups=[{'GENDER': 0}], privileged_groups=[{'GENDER': 1}])
reweighed_dataset = rw.fit_transform(dataset)

# Check bias metrics before and after
metric_orig = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{'GENDER': 0}])
metric_new = BinaryLabelDatasetMetric(reweighed_dataset, unprivileged_groups=[{'GENDER': 0}])

print(f"Original Disparate Impact: {metric_orig.disparate_impact()}")
print(f"Reweighed Disparate Impact: {metric_new.disparate_impact()}")