from .celery_app import celery_app
import joblib
import pandas as pd
from io import BytesIO
import base64
import json
from fairness_troops import BiasAuditor, BiasExplainer, ReportGenerator, visuals

@celery_app.task(bind=True)
def run_audit_task(self, model_bytes_b64: str, data_csv_str: str, config: dict):
    """
    Async task to run audit, generate explanations, and create a report.
    """
    try:
        self.update_state(state='PROGRESS', meta={'message': 'Loading data...'})
        
        # Decode inputs
        model = joblib.load(BytesIO(base64.b64decode(model_bytes_b64)))
        data = pd.read_csv(BytesIO(data_csv_str.encode('utf-8')))
        
        target_col = config['target_col']
        sensitive_col = config['sensitive_col']
        privileged_group = config.get('privileged_group')
        unprivileged_group = config.get('unprivileged_group')

        # 1. Run Fairness Audit
        self.update_state(state='PROGRESS', meta={'message': 'Calculating fairness metrics...'})
        auditor = BiasAuditor(
            model=model,
            dataset=data,
            target_col=target_col,
            sensitive_col=sensitive_col,
            privileged_group=privileged_group,
            unprivileged_group=unprivileged_group
        )
        report = auditor.run_audit()
        predictions = auditor.y_pred.tolist()
        mitigation_weights = auditor.get_mitigation_weights().tolist()

        # 2. Generate Explanations (Permutation Importance & PDP)
        self.update_state(state='PROGRESS', meta={'message': 'Generating explanations (Permutation & PDP)...'})
        
        # Prepare data for explanation
        # Ideally we use a held-out set, but here we use the audit dataset (acting as test set)
        X_test = data.drop(columns=[target_col])
        y_test = data[target_col]
        
        # Initialize Explainer
        explainer = BiasExplainer(model, X_test, y_test)
        
        # Generate Permutation Importance
        perm_importance_b64 = explainer.generate_permutation_importance_plot()
        
        # Generate PDP for Sensitive Column and maybe top 2 others (if we calculated top features)
        # For simplicity, let's just plot the sensitive column and the first column to show usage
        pdp_features = [sensitive_col]
        if len(X_test.columns) > 1:
            other_feat = [c for c in X_test.columns if c != sensitive_col][0]
            pdp_features.append(other_feat)
            
        pdp_plot_b64 = explainer.generate_pdp_plot(pdp_features)
        
        # 3. Generate Visuals (Standard Fairness Plots)
        self.update_state(state='PROGRESS', meta={'message': 'Generating fairness plots...'})
        visual_report = auditor.get_visuals()
        visual_b64s = {}
        
        # Convert matplotlib figures to base64 for the PDF report
        for name, fig in visual_report.items():
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            visual_b64s[name] = base64.b64encode(buf.read()).decode('utf-8')
        
        # Add Explanations to visuals
        if perm_importance_b64:
            visual_b64s['feature_importance'] = perm_importance_b64
        if pdp_plot_b64:
            visual_b64s['pdp_plot'] = pdp_plot_b64

        # 4. Generate PDF Report
        self.update_state(state='PROGRESS', meta={'message': 'Generating PDF report...'})
        pdf_gen = ReportGenerator(report, config)
        pdf_bytes = pdf_gen.generate(visual_b64s)
        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')

        return {
            "status": "success",
            "metrics": report,
            "predictions": predictions,
            "mitigation_weights": mitigation_weights,
            "pdf_report_b64": pdf_b64,
            "visuals": visual_b64s # Optional: send images back to frontend if needed
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Custom error handling to propagate message
        return {"status": "error", "error": str(e)}
