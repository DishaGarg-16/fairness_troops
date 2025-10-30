# src/bias_debugger/metrics.py
import pandas as pd

def calculate_disparate_impact(
    y_pred: pd.Series,
    sensitive_features: pd.Series,
    privileged_group: str | int,
    unprivileged_group: str | int,
) -> float:
    """
    Calculates the Disparate Impact Ratio.
    Ratio of favorable outcomes for unprivileged vs. privileged groups.
    A value < 0.8 or > 1.2 is often considered unfair. Ideal is 1.0.
    """
    try:
        # % of unprivileged group that received a favorable outcome (y_pred=1)
        unprivileged_mask = sensitive_features == unprivileged_group
        unprivileged_rate = y_pred[unprivileged_mask].mean()
        
        # % of privileged group that received a favorable outcome (y_pred=1)
        privileged_mask = sensitive_features == privileged_group
        privileged_rate = y_pred[privileged_mask].mean()
        
        if privileged_rate == 0:
            # Avoid division by zero. If priv rate is 0,
            # DI is undefined or 1.0 if unpriv rate is also 0.
            return 1.0 if unprivileged_rate == 0 else float('inf')
            
        return unprivileged_rate / privileged_rate
    
    except Exception as e:
        print(f"Error in calculate_disparate_impact: {e}")
        return float('nan')

def calculate_equal_opportunity_difference(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.Series,
    privileged_group: str | int,
    unprivileged_group: str | int,
) -> float:
    """
    Calculates the Equal Opportunity Difference.
    Difference in True Positive Rates (TPR) between groups.
    TPR_unprivileged - TPR_privileged. Ideal is 0.
    """
    try:
        # Privileged group masks
        priv_mask = sensitive_features == privileged_group
        priv_true_positives = (y_true == 1) & (y_pred == 1) & priv_mask
        priv_all_positives = (y_true == 1) & priv_mask
        
        tpr_privileged = priv_true_positives.sum() / priv_all_positives.sum()
        if priv_all_positives.sum() == 0:
            tpr_privileged = 0.0

        # Unprivileged group masks
        unpriv_mask = sensitive_features == unprivileged_group
        unpriv_true_positives = (y_true == 1) & (y_pred == 1) & unpriv_mask
        unpriv_all_positives = (y_true == 1) & unpriv_mask
        
        tpr_unprivileged = unpriv_true_positives.sum() / unpriv_all_positives.sum()
        if unpriv_all_positives.sum() == 0:
            tpr_unprivileged = 0.0
            
        return tpr_unprivileged - tpr_privileged
    
    except Exception as e:
        print(f"Error in calculate_equal_opportunity_difference: {e}")
        return float('nan')