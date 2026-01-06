import shap
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd

class BiasExplainer:
    """
    Generates global and local explanations for the model using SHAP.
    """
    def __init__(self, model, X_train: pd.DataFrame):
        """
        :param model: The trained model.
        :param X_train: Background dataset for SHAP (subset of training data).
        """
        self.model = model
        self.X_train = X_train
        
        # Determine explainer type
        # For tree-based models (RandomForest, XGBoost, etc.), use TreeExplainer
        # For linear models, LinearExplainer
        # Fallback to KernelExplainer (slow)
        
        try:
            # Simple heuristic
            model_name = type(model).__name__.lower()
            if 'forest' in model_name or 'xgb' in model_name or 'tree' in model_name:
                self.explainer = shap.TreeExplainer(model)
            elif 'linear' in model_name or 'logistic' in model_name:
                 self.explainer = shap.LinearExplainer(model, X_train)
            else:
                 # Summarize background for KernelExplainer to speed up
                 self.background = shap.kmeans(X_train, 10) if len(X_train) > 100 else X_train
                 self.explainer = shap.KernelExplainer(model.predict, self.background)
                 
        except Exception as e:
            print(f"Error initializing specific explainer, defaulting to KernelExplainer: {e}")
            self.background = shap.kmeans(X_train, 10) if len(X_train) > 100 else X_train
            self.explainer = shap.KernelExplainer(model.predict, self.background)

    def generate_global_importance_plot(self, X_sample) -> str:
        """
        Generates a SHAP summary plot as a base64 encoded image.
        """
        try:
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle list of shap values (e.g., for classification outputs)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Take positive class
                
            plt.figure()
            shap.summary_plot(shap_values, X_sample, show=False)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            print(f"Error generating global plot: {e}")
            return ""

    def get_shap_values(self, X_sample):
        """
        Returns raw SHAP values for further processing/reporting.
        """
        try:
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            return shap_values
        except Exception as e:
             print(f"Error calcuating SHAP values: {e}")
             return None
