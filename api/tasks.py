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

        # 2. Generate SHAP Explanations
        self.update_state(state='PROGRESS', meta={'message': 'Generating SHAP explanations (this may take a while)...'})
        # Use a sample for SHAP to speed up explainability (e.g. 100 samples)
        X_test = data.drop(columns=[target_col])
        X_sample = X_test.sample(min(100, len(X_test)))
        
        explainer = BiasExplainer(model, X_test.head(min(50, len(X_test)))) # Use small background
        shap_plot_b64 = explainer.generate_global_importance_plot(X_sample)
        
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
        
        # Add SHAP to visuals
        if shap_plot_b64:
            visual_b64s['shap_summary'] = shap_plot_b64

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
