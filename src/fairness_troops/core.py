# src/bias_debugger/core.py
import pandas as pd
from . import metrics
from . import visuals
from . import mitigation

class BiasAuditor:
    """
    The main class for auditing a model for bias.
    """
    def __init__(
        self,
        model,
        dataset: pd.DataFrame,
        target_col: str,
        sensitive_col: str,
        privileged_group: str | int,
        unprivileged_group: str | int
    ):
        """
        Initializes the auditor.
        
        :param model: A trained, scikit-learn compatible model (with .predict())
        :param dataset: The test dataset (DataFrame)
        :param target_col: The name of the true label column (e.g., 'loan_approved')
        :param sensitive_col: The name of the sensitive attribute (e.g., 'gender')
        :param privileged_group: The value for the privileged group (e.g., 'Male')
        :param unprivileged_group: The value for the unprivileged group (e.g., 'Female')
        """
        self.model = model
        self.dataset = dataset
        self.target_col = target_col
        self.sensitive_col = sensitive_col
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group
        
        # Prepare data for model
        self.y_true = dataset[target_col]
        # X_test contains all columns EXCEPT the target
        self.X_test = dataset.drop(columns=[target_col])
        self.sensitive_features = dataset[sensitive_col]
        
        # Get model predictions
        # We wrap in pd.Series to ensure it has the same index as y_true
        self.y_pred = pd.Series(
            model.predict(self.X_test), 
            index=self.y_true.index, 
            name="predictions"
        )
        
        # Store results
        self.report = {}
        self.visual_report = {}

    def run_audit(self) -> dict:
        """
        Runs all fairness metrics and returns a report dictionary.
        """
        self.report['disparate_impact'] = metrics.calculate_disparate_impact(
            self.y_pred,
            self.sensitive_features,
            self.privileged_group,
            self.unprivileged_group
        )
        
        self.report['equal_opportunity_diff'] = metrics.calculate_equal_opportunity_difference(
            self.y_true,
            self.y_pred,
            self.sensitive_features,
            self.privileged_group,
            self.unprivileged_group
        )
        
        return self.report

    def get_visuals(self) -> dict:
        """
        Generates all fairness visualizations and returns a dictionary of Figures.
        """
        self.visual_report['group_outcomes'] = visuals.plot_group_outcomes(
            self.y_pred,
            self.sensitive_features,
            title=f"Favorable Outcome Rate ({self.target_col}=1)"
        )
        
        self.visual_report['tpr_by_group'] = visuals.plot_tpr_by_group(
            self.y_true,
            self.y_pred,
            self.sensitive_features,
            title=f"True Positive Rate by {self.sensitive_col}"
        )
        
        return self.visual_report
        
    def get_mitigation_weights(self) -> pd.Series:
        """
        Calculates reweighting weights as a mitigation strategy.
        Note: This should be applied to the *training* data, but we
        calculate it here based on the test set distribution as an example.
        A better implementation would take training data as input.
        """
        
        # Create a dataframe with just the target and sensitive columns
        # This is what the mitigation function needs
        mitigation_df = self.dataset[[self.target_col, self.sensitive_col]]
        
        weights = mitigation.get_reweighting_weights(
            mitigation_df,
            self.target_col,
            self.sensitive_col
        )
        weights.name = "sample_weight"
        return weights