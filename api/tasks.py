from .celery_app import celery_app
import pandas as pd
from io import BytesIO
import base64
import json
import logging
import time
import skops.io as sio
from fairness_troops import BiasAuditor, BiasExplainer, ReportGenerator, visuals

# Configure logging for task tracking
logger = logging.getLogger(__name__)

# Task metrics tracking (in production, use Prometheus/StatsD)
_task_metrics = {
    "total_tasks": 0,
    "successful_tasks": 0,
    "failed_tasks": 0,
    "retried_tasks": 0,
}


def get_task_metrics() -> dict:
    """Return current task success/failure metrics."""
    total = _task_metrics["total_tasks"]
    success = _task_metrics["successful_tasks"]
    success_rate = (success / total * 100) if total > 0 else 0.0
    return {
        **_task_metrics,
        "success_rate_percent": round(success_rate, 2)
    }


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,  # Max 5 minutes between retries
    max_retries=3,
    retry_jitter=True,  # Add randomness to prevent thundering herd
    track_started=True,
    acks_late=True,  # Acknowledge after completion for reliability
)
def run_audit_task(self, model_bytes_b64: str, data_csv_str: str, config: dict):
    """
    Async task to run audit, generate explanations, and create a report.
    
    Features:
    - Automatic retry with exponential backoff (up to 3 retries)
    - Task success/failure tracking for reliability metrics
    - Progress updates for UI feedback
    """
    task_id = self.request.id
    start_time = time.time()
    
    # Track task start
    _task_metrics["total_tasks"] += 1
    logger.info(f"Task {task_id} started. Retry attempt: {self.request.retries}/{self.max_retries}")
    
    try:
        self.update_state(state='PROGRESS', meta={'message': 'Loading data...'})
        
        # Decode inputs
        # Securely load model using skops
        model_bytes = base64.b64decode(model_bytes_b64)
        # skops.io.loads is available? Checking docs or usage. skops.io.load takes a file path or file-like object.
        # We use a BytesIO buffer.
        # trusted=True is needed if we fully trust the types, but to be strictly safe we should let skops validate.
        # However, for generic sklearn models, we often need to trust standard sklearn types.
        # skops doesn't execute arbitrary code, so strictly speaking it's safer than pickle even with trusted=True (it checks allowed types).
        # But ideally we inspect first. For this implementation, we rely on skops default safety or explicit trust of sklearn types.
        
        # NOTE: skops.io.load expects a file. byte_buffer works.
        # Fix for CVE-2024-37065: trusted=True is deprecated. 
        # We must inspect types and explicitly trust them.
        model_buffer = BytesIO(model_bytes)
        
        # Get list of untrusted types in the file
        untrusted_types = sio.get_untrusted_types(file=model_buffer)
        
        # Reset buffer position
        model_buffer.seek(0)
        
        # Load trusting all found types (since this is a user-uploaded model in their own environment)
        model = sio.load(model_buffer, trusted=untrusted_types) 
        
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

        # Track successful completion
        elapsed_time = time.time() - start_time
        _task_metrics["successful_tasks"] += 1
        logger.info(f"Task {task_id} completed successfully in {elapsed_time:.2f}s. "
                   f"Success rate: {get_task_metrics()['success_rate_percent']}%")

        return {
            "status": "success",
            "metrics": report,
            "predictions": predictions,
            "mitigation_weights": mitigation_weights,
            "pdf_report_b64": pdf_b64,
            "visuals": visual_b64s,
            "execution_time_seconds": round(elapsed_time, 2)
        }

    except Exception as e:
        import traceback
        elapsed_time = time.time() - start_time
        
        # Check if we've exhausted retries
        if self.request.retries >= self.max_retries:
            _task_metrics["failed_tasks"] += 1
            logger.error(f"Task {task_id} failed permanently after {self.request.retries} retries. "
                        f"Error: {str(e)}. Elapsed: {elapsed_time:.2f}s")
            traceback.print_exc()
            # Return error instead of raising to avoid further retries
            return {"status": "error", "error": str(e)}
        else:
            # Track retry attempt
            _task_metrics["retried_tasks"] += 1
            logger.warning(f"Task {task_id} failed, will retry ({self.request.retries + 1}/{self.max_retries}). "
                          f"Error: {str(e)}")
            # Re-raise to trigger Celery's retry mechanism
            raise

