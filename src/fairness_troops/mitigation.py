# src/bias_debugger/mitigation.py
import pandas as pd

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