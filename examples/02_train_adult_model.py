import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import skops.io as sio
import os

print("Fetching Adult Census Income dataset from OpenML...")
try:
    # Fetch dataset (ID 45068 is what usually maps to 'adult')
    # as_frame=True returns pandas DataFrame
    data = fetch_openml(data_id=45068, as_frame=True, parser='auto')
    X = data.data
    y = data.target
except Exception as e:
    print(f"Error fetching data: {e}")
    print("Falling back to name search...")
    data = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
    X = data.data
    y = data.target

print(f"Dataset shape: {X.shape}")

# Preprocessing
# Convert target to 0/1 (<=50K is usually 0, >50K is 1)
# Note: The raw target might be boolean or strings like '<=50K', '>50K', '<=50K.' etc.
print("Target values before encoding:", y.unique())
y = y.astype(str).apply(lambda x: 1 if '>50K' in x else 0)
print("Target distribution:", y.value_counts())

# Select a subset of columns for simplicity and speed
# We want 'sex' and 'race' for fairness auditing
selected_features = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'hours-per-week']
X = X[selected_features]

# Handle Missing Values (simple drop for this example)
t = pd.concat([X, y], axis=1).dropna()
X = t[selected_features]
y = t.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define column types
numeric_features = ['age', 'hours-per-week']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']

# Create preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    remainder='drop'
)

# Create pipeline
model = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=1000) # Increased max_iter for convergence
)

print("Training model...")
model.fit(X_train, y_train)

# accuracy
score = model.score(X_test, y_test)
print(f"Test Accuracy: {score:.4f}")

# Save Model
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, 'adult_model.skops')
sio.dump(model, model_path)
print(f"Model saved to {model_path}")

# Save a sample of test data (e.g., 500 rows) for the user to download/use
# We save the features AND the target (renamed to 'income' for clarity or kept as is)
sample_data = X_test.copy()
sample_data['income'] = y_test # Target column
sample_data = sample_data.sample(n=1000, random_state=42)

data_path = os.path.join(output_dir, 'adult_test_data.csv')
sample_data.to_csv(data_path, index=False)
print(f"Sample test data saved to {data_path}")

print("Done!")
