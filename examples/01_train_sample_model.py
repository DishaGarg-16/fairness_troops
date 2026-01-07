# examples/01_train_sample_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import skops.io as sio
import os

print("Training sample model...")

# Define file paths
DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_loan_data.csv')
MODEL_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_model.skops')

# Load data
df = pd.read_csv(DATA_FILE)

# Define features (X) and target (y)
FEATURES = ['age', 'income', 'credit_score', 'gender']
TARGET = 'loan_approved'

X = df[FEATURES]
y = df[TARGET]

# Create a preprocessor
# We need to One-Hot Encode 'gender' so the model can read it
preprocessor = make_column_transformer(
    (OneHotEncoder(drop='if_binary'), ['gender']),
    remainder='passthrough'
)

# Create a simple logistic regression model
model = make_pipeline(
    preprocessor,
    LogisticRegression()
)

# Train the model
model.fit(X, y)

# Save the model
sio.dump(model, MODEL_FILE)

print(f"Model saved to {MODEL_FILE}")
print("You can now run the Streamlit app and upload this model and the CSV data.")