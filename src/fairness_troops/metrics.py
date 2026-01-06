# src/bias_debugger/metrics.py
import pandas as pd
import numpy as np
from fairlearn.metrics import (
    demographic_parity_ratio,
    demographic_parity_difference,
    equal_opportunity_difference,
    false_positive_rate,
    false_negative_rate,
    MetricFrame
)
from sklearn.metrics import confusion_matrix

def calculate_disparate_impact(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.Series,
) -> float:
    """
    Calculates the Disparate Impact Ratio (Demographic Parity Ratio).
    Ratio of favorable outcomes for unprivileged vs. privileged groups.
    """
    try:
        # fairlearn handles the group identification if we pass sensitive_features
        # We assume sensitive_features is correctly identifying groups
        return demographic_parity_ratio(
            y_true,
            y_pred,
            sensitive_features=sensitive_features
        )
    except Exception as e:
        print(f"Error in calculate_disparate_impact: {e}")
        return float('nan')

def calculate_statistical_parity_difference(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.Series,
) -> float:
    """
    Calculates Statistical Parity Difference (Demographic Parity Difference).
    Difference in positive outcome rates between groups.
    """
    try:
        return demographic_parity_difference(
            y_true,
            y_pred,
            sensitive_features=sensitive_features
        )
    except Exception as e:
        print(f"Error in calculate_statistical_parity_difference: {e}")
        return float('nan')

def calculate_equal_opportunity_difference(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.Series,
) -> float:
    """
    Calculates the Equal Opportunity Difference.
    Difference in True Positive Rates (TPR) between groups.
    """
    try:
        return equal_opportunity_difference(
            y_true,
            y_pred,
            sensitive_features=sensitive_features
        )
    except Exception as e:
        print(f"Error in calculate_equal_opportunity_difference: {e}")
        return float('nan')

def calculate_false_positive_rate_difference(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.Series,
) -> float:
    """
    Calculates the difference in False Positive Rates between groups.
    """
    try:
        # We use MetricFrame to get FPR for each group then take the difference
        mf = MetricFrame(
            metrics=false_positive_rate,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        return mf.difference()
    except Exception as e:
        print(f"Error in calculate_false_positive_rate_difference: {e}")
        return float('nan')

def calculate_false_negative_rate_difference(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.Series,
) -> float:
    """
    Calculates the difference in False Negative Rates between groups.
    """
    try:
        mf = MetricFrame(
            metrics=false_negative_rate,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        return mf.difference()
    except Exception as e:
        print(f"Error in calculate_false_negative_rate_difference: {e}")
        return float('nan')

def calculate_average_abs_odds_difference(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.Series,
) -> float:
    """
    Calculates Average Absolute Odds Difference.
    0.5 * (|FPR_diff| + |TPR_diff|)
    """
    try:
        fpr_diff = calculate_false_positive_rate_difference(y_true, y_pred, sensitive_features)
        tpr_diff = calculate_equal_opportunity_difference(y_true, y_pred, sensitive_features) # Equal Opp Diff IS TPR diff
        
        return 0.5 * (abs(fpr_diff) + abs(tpr_diff))
    except Exception as e:
        print(f"Error in calculate_average_abs_odds_difference: {e}")
        return float('nan')

def calculate_theil_index(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Calculates the Theil Index (Generalized Entropy Index with alpha=1).
    Used for individual fairness. 
    Here we apply it to b = 1 + y_pred - y_true (benefit) to handle 0s, 
    following standard implementations for classification.
    """
    try:
        # Convert to numpy for easier math
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Simple benefit vector: 
        # For binary classification, this might not be perfect but is a standard placeholder
        # Ideally y_pred is probability. If binary, b takes values 0, 1, 2.
        b = 1 + y_pred - y_true 
        
        mean_b = np.mean(b)
        if mean_b == 0:
            return 0.0
            
        # Theil index formula: (1/n) * sum( (b_i / mean_b) * ln(b_i / mean_b) )
        # Handle division by zero or log of zero/negative by replacing with 1 (log(1)=0) where approx 0
        
        return np.mean((b / mean_b) * np.log((b / mean_b) + 1e-10))
    except Exception as e:
        print(f"Error in calculate_theil_index: {e}")
        return float('nan')
