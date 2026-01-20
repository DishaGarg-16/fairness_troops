# src/bias_debugger/mitigation.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def get_reweighting_weights(
    dataset: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
) -> pd.Series:
    """
    Calculates sample weights to mitigate bias using reweighting.
    The formula gives more weight to under-represented group/outcome combinations
    and less weight to over-represented ones.
    
    Weight = P(Group) * P(Outcome) / P(Group, Outcome)
    """
    
    # P(Group, Outcome) - Joint probability
    joint_prob = pd.crosstab(
        dataset[sensitive_col], dataset[target_col], normalize='all'
    )
    
    # P(Group) - Marginal probability for sensitive feature
    prob_group = dataset[sensitive_col].value_counts(normalize=True)
    
    # P(Outcome) - Marginal probability for target
    prob_outcome = dataset[target_col].value_counts(normalize=True)
    
    # Calculate weights
    weights = pd.Series(index=dataset.index, dtype=float)
    
    for group in prob_group.index:
        for outcome in prob_outcome.index:
            # Find all rows matching this group and outcome
            mask = (dataset[sensitive_col] == group) & (dataset[target_col] == outcome)
            
            # Get joint probability P(group, outcome)
            p_joint = joint_prob.loc[group, outcome]
            
            if p_joint > 0:
                # Weight = P(group) * P(outcome) / P(group, outcome)
                weight_val = (prob_group[group] * prob_outcome[outcome]) / p_joint
                weights[mask] = weight_val
            else:
                weights[mask] = 0 # This case shouldn't happen if data is well-formed
                
    return weights


def apply_threshold_optimizer(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_features_train: pd.Series,
    X_test: pd.DataFrame,
    sensitive_features_test: pd.Series,
    constraint: str = "equalized_odds",
    objective: str = "balanced_accuracy_score",
) -> Tuple[np.ndarray, object]:
    """
    Post-processing technique using Fairlearn's ThresholdOptimizer.
    Adjusts decision thresholds per group to satisfy fairness constraints.
    
    :param estimator: A trained sklearn-compatible estimator with predict_proba
    :param X_train: Training features (used to fit the optimizer)
    :param y_train: Training labels
    :param sensitive_features_train: Sensitive attribute for training data
    :param X_test: Test features to generate fair predictions for
    :param sensitive_features_test: Sensitive attribute for test data
    :param constraint: Fairness constraint - 'equalized_odds' or 'demographic_parity'
    :param objective: Objective to optimize - 'balanced_accuracy_score' or 'accuracy_score'
    :return: Tuple of (fair predictions array, fitted ThresholdOptimizer)
    """
    from fairlearn.postprocessing import ThresholdOptimizer
    
    # Create and fit the threshold optimizer
    postprocess_est = ThresholdOptimizer(
        estimator=estimator,
        constraints=constraint,
        objective=objective,
        prefit=True,  # We pass a pre-fitted estimator
        predict_method="predict_proba"
    )
    
    postprocess_est.fit(X_train, y_train, sensitive_features=sensitive_features_train)
    
    # Generate fair predictions
    y_pred_fair = postprocess_est.predict(X_test, sensitive_features=sensitive_features_test)
    
    return y_pred_fair, postprocess_est


def apply_reject_option_classification(
    y_pred_proba: np.ndarray,
    sensitive_features: pd.Series,
    threshold: float = 0.5,
    margin: float = 0.1,
    favorable_label: int = 1,
) -> np.ndarray:
    """
    Post-processing technique: Reject Option Classification (ROC).
    Flips predictions near the decision boundary to improve fairness.
    
    For samples in the "critical region" (threshold ± margin):
    - Unprivileged group members with unfavorable predictions -> flip to favorable
    - Privileged group members with favorable predictions -> flip to unfavorable
    
    :param y_pred_proba: Probability predictions (for positive class)
    :param sensitive_features: Series indicating group membership
    :param threshold: Decision threshold (default 0.5)
    :param margin: Width of critical region on each side of threshold
    :param favorable_label: The label considered favorable (default 1)
    :return: Adjusted binary predictions
    """
    y_pred_proba = np.array(y_pred_proba)
    sensitive_features = np.array(sensitive_features)
    
    # Initial predictions based on threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Identify critical region
    lower_bound = threshold - margin
    upper_bound = threshold + margin
    in_critical_region = (y_pred_proba >= lower_bound) & (y_pred_proba <= upper_bound)
    
    # Identify privileged group (the one with higher positive rate in original predictions)
    groups = np.unique(sensitive_features)
    if len(groups) != 2:
        # ROC works best with binary groups; return original for multi-group
        return y_pred
    
    group_positive_rates = {}
    for g in groups:
        mask = sensitive_features == g
        group_positive_rates[g] = y_pred[mask].mean()
    
    privileged_group = max(group_positive_rates, key=group_positive_rates.get)
    unprivileged_group = min(group_positive_rates, key=group_positive_rates.get)
    
    # Apply ROC adjustments
    # For unprivileged in critical region with unfavorable prediction -> flip to favorable
    unpriv_unfavorable_critical = (
        in_critical_region & 
        (sensitive_features == unprivileged_group) & 
        (y_pred != favorable_label)
    )
    y_pred[unpriv_unfavorable_critical] = favorable_label
    
    # For privileged in critical region with favorable prediction -> flip to unfavorable
    priv_favorable_critical = (
        in_critical_region & 
        (sensitive_features == privileged_group) & 
        (y_pred == favorable_label)
    )
    y_pred[priv_favorable_critical] = 1 - favorable_label
    
    return y_pred


def calculate_fairness_improvement(
    y_true: pd.Series,
    y_pred_original: pd.Series,
    y_pred_mitigated: pd.Series,
    sensitive_features: pd.Series,
) -> dict:
    """
    Calculate the improvement in fairness metrics after mitigation.
    
    :return: Dict with original, mitigated values and percentage improvement
    """
    from . import metrics
    
    # Calculate original metrics
    original_spd = abs(metrics.calculate_statistical_parity_difference(
        y_true, y_pred_original, sensitive_features
    ))
    original_eod = abs(metrics.calculate_equal_opportunity_difference(
        y_true, y_pred_original, sensitive_features
    ))
    original_di = metrics.calculate_disparate_impact(
        y_true, y_pred_original, sensitive_features
    )
    
    # Calculate mitigated metrics
    mitigated_spd = abs(metrics.calculate_statistical_parity_difference(
        y_true, y_pred_mitigated, sensitive_features
    ))
    mitigated_eod = abs(metrics.calculate_equal_opportunity_difference(
        y_true, y_pred_mitigated, sensitive_features
    ))
    mitigated_di = metrics.calculate_disparate_impact(
        y_true, y_pred_mitigated, sensitive_features
    )
    
    # Calculate improvements (reduction in gap)
    spd_improvement = ((original_spd - mitigated_spd) / original_spd * 100) if original_spd > 0 else 0
    eod_improvement = ((original_eod - mitigated_eod) / original_eod * 100) if original_eod > 0 else 0
    
    # DI improvement: closer to 1.0 is better
    di_gap_original = abs(1.0 - original_di)
    di_gap_mitigated = abs(1.0 - mitigated_di)
    di_improvement = ((di_gap_original - di_gap_mitigated) / di_gap_original * 100) if di_gap_original > 0 else 0
    
    return {
        "statistical_parity_difference": {
            "original": original_spd,
            "mitigated": mitigated_spd,
            "improvement_percent": spd_improvement
        },
        "equal_opportunity_difference": {
            "original": original_eod,
            "mitigated": mitigated_eod,
            "improvement_percent": eod_improvement
        },
        "disparate_impact": {
            "original": original_di,
            "mitigated": mitigated_di,
            "improvement_percent": di_improvement
        }
    }