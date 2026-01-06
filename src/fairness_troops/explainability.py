import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

class BiasExplainer:
    """
    Generates explanations for the model using Permutation Importance and PDP.
    Replaces SHAP due to compatibility issues.
    """
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.Series = None):
        """
        :param model: The trained model.
        :param X_train: Training features (needed for permutation importance).
        :param y_train: Training labels (needed for scoring permutation importance), optional. 
                        If not provided, importance is calculated based on model score only which might need labels.
                        For simplicity in audit, we might use the test set passed as 'dataset' if y is available.
        """
        self.model = model
        self.X_data = X_train
        self.y_data = y_train

    def generate_permutation_importance_plot(self) -> str:
        """
        Generates a Permutation Feature Importance plot.
        """
        try:
            # Calculate permutation importance
            # Note: We need y_true to calculate importance based on accuracy/score drop.
            # If y_data is not provided, we can't strictly calculate "importance" defined as drop in score
            # unless the model has an unsupervised score method (unlikely).
            # Assuming y_data is passed or we default to the model's predict behavior if compatible? 
            # Actually sklearn permutation_importance REQUIRES y.
            
            if self.y_data is None:
                return ""

            result = permutation_importance(
                self.model, self.X_data, self.y_data, n_repeats=10, random_state=42, n_jobs=-1
            )
            
            sorted_idx = result.importances_mean.argsort()
            
            plt.figure(figsize=(10, 6))
            plt.boxplot(
                result.importances[sorted_idx].T,
                vert=False,
                labels=self.X_data.columns[sorted_idx]
            )
            plt.title("Permutation Importances (test set)")
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            print(f"Error generating permutation importance plot: {e}")
            return ""

    def generate_pdp_plot(self, features: list) -> str:
        """
        Generates Partial Dependence Plots for top features.
        :param features: List of feature names or indices to plot.
        """
        try:
            # We assume features are valid column names or indices
            # Limit to top 3 to avoid overcrowding
            features_to_plot = features[:3]
            
            fig, ax = plt.subplots(figsize=(12, 4))
            PartialDependenceDisplay.from_estimator(
                self.model, 
                self.X_data, 
                features_to_plot,
                ax=ax,
                kind='average'
            )
            plt.suptitle(f"Partial Dependence Plot for {features_to_plot}", y=1.02)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')

        except Exception as e:
            print(f"Error generating PDP plot: {e}")
            return ""
