import pandas as pd
import numpy as np
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from fairness_troops import BiasAuditor
from sklearn.linear_model import LogisticRegression

# 1. Create Synthetic Data
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 100),
    'feature2': np.random.normal(0, 1, 100),
    'gender': np.random.choice(['Male', 'Female'], 100), # Sensitive
    'approved': np.random.choice([0, 1], 100) # Target
})

# 2. Train a dummy model
# We need numeric data for sklearn LogisticRegression
data['gender_numeric'] = data['gender'].map({'Male': 0, 'Female': 1})
X = data[['feature1', 'feature2', 'gender_numeric']] 
y = data['approved']
model = LogisticRegression()
model.fit(X, y)

# 3. Run Auditor
print("Running BiasAuditor...")
# For the auditor, we pass the original dataframe but we need to ensure X_test passed to predict matches X
# The current BiasAuditor implementation drops target_col and passes the rest to predict.
# So we need to construct a dataset that has target + features model expects.
# Our model expects feature1, feature2, gender_numeric.
# Our sensitive col will be 'gender' (original) or 'gender_numeric'. 
# Let's use 'gender_numeric' as sensitive_col for simplicity in this test to avoid re-writing BiasAuditor logic 
# (typically one would robustly handle this column matching).

dataset_for_audit = data[['feature1', 'feature2', 'gender_numeric', 'approved']].copy()

auditor = BiasAuditor(
    model=model,
    dataset=dataset_for_audit,
    target_col='approved',
    sensitive_col='gender_numeric',
    privileged_group=0, # Male
    unprivileged_group=1 # Female
)

report = auditor.run_audit()

# 4. Print Results
print("\n--- Fairness Metrics Report ---")
for metric, value in report.items():
    print(f"{metric}: {value}")

# 5. Assertions
expected_keys = [
    'disparate_impact', 'equal_opportunity_diff', 'avg_abs_odds_diff',
    'theil_index', 'statistical_parity_diff', 
    'false_positive_rate_diff', 'false_negative_rate_diff'
]

missing = [k for k in expected_keys if k not in report]
if missing:
    print(f"FAILED: Missing keys: {missing}")
    sys.exit(1)

print("\nVerification Complete: All keys present.")
