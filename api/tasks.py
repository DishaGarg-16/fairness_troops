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

# Maximum number of unique values for a column to be considered categorical
MAX_CATEGORICAL_UNIQUE = 20


class InputValidationError(Exception):
    """Raised for user-input validation failures. Must NOT trigger Celery retries."""
    pass


def validate_audit_inputs(model, data: pd.DataFrame, config: dict) -> None:
    """
    Lightweight validation of user inputs before any heavy processing.
    Raises InputValidationError with a user-friendly message on failure.
    """
    target_col = config['target_col']
    sensitive_col = config['sensitive_col']
    privileged_group = config.get('privileged_group')
    unprivileged_group = config.get('unprivileged_group')

    # 1. Check that required columns exist in the dataset
    if target_col not in data.columns:
        raise InputValidationError(
            f"Target column '{target_col}' not found in the dataset. "
            f"Available columns: {list(data.columns)}"
        )
    if sensitive_col not in data.columns:
        raise InputValidationError(
            f"Sensitive column '{sensitive_col}' not found in the dataset. "
            f"Available columns: {list(data.columns)}"
        )

    # 2. Target variable must be binary (exactly 2 unique values)
    target_unique = data[target_col].nunique()
    if target_unique != 2:
        raise InputValidationError(
            f"Target column '{target_col}' must be binary (exactly 2 unique values), "
            f"but found {target_unique} unique values. "
            f"Fairness metrics require binary classification targets."
        )

    # 3. Sensitive column must be categorical (reasonable number of groups)
    sensitive_unique = data[sensitive_col].nunique()
    if sensitive_unique > MAX_CATEGORICAL_UNIQUE:
        raise InputValidationError(
            f"Sensitive column '{sensitive_col}' has {sensitive_unique} unique values, "
            f"which looks like a continuous variable rather than a categorical attribute. "
            f"Please select a categorical column with at most {MAX_CATEGORICAL_UNIQUE} unique groups."
        )
    if sensitive_unique < 2:
        raise InputValidationError(
            f"Sensitive column '{sensitive_col}' must have at least 2 unique groups, "
            f"but found only {sensitive_unique}."
        )

    # 4. Check that privileged/unprivileged groups actually exist in the sensitive column
    sensitive_values = data[sensitive_col].astype(str).unique().tolist()
    if privileged_group and str(privileged_group) not in sensitive_values:
        raise InputValidationError(
            f"Privileged group '{privileged_group}' not found in column '{sensitive_col}'. "
            f"Available groups: {sensitive_values}"
        )
    if unprivileged_group and str(unprivileged_group) not in sensitive_values:
        raise InputValidationError(
            f"Unprivileged group '{unprivileged_group}' not found in column '{sensitive_col}'. "
            f"Available groups: {sensitive_values}"
        )

    # 5. Feature count matching — check model expects the right number of features
    X_test = data.drop(columns=[target_col])
    n_dataset_features = len(X_test.columns)

    if hasattr(model, 'n_features_in_'):
        n_model_features = model.n_features_in_
        if n_dataset_features != n_model_features:
            raise InputValidationError(
                f"Feature count mismatch: the model expects {n_model_features} features, "
                f"but the dataset has {n_dataset_features} features (after dropping the target column). "
                f"Please upload a dataset that matches the model's training features."
            )

    # 6. Feature name matching (if available) — gives more specific error messages
    if hasattr(model, 'feature_names_in_'):
        model_features = set(model.feature_names_in_)
        data_features = set(X_test.columns)
        missing_in_data = model_features - data_features
        extra_in_data = data_features - model_features
        if missing_in_data or extra_in_data:
            msg = "Feature name mismatch between model and dataset."
            if missing_in_data:
                msg += f" Missing from dataset: {sorted(missing_in_data)}."
            if extra_in_data:
                msg += f" Extra in dataset (not expected by model): {sorted(extra_in_data)}."
            raise InputValidationError(msg)

    logger.info("Input validation passed.")

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
    # InputValidationError is excluded from auto-retry because it's a user-input
    # error that will always fail — retrying is wasteful.
    dont_autoretry_for=(InputValidationError,),
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
        model_bytes = base64.b64decode(model_bytes_b64)
        
        # Securely load model using skops
        # Fix for CVE-2024-37065: trusted=True is deprecated. 
        # We must inspect types and explicitly trust them.
        model_buffer = BytesIO(model_bytes)
        untrusted_types = sio.get_untrusted_types(file=model_buffer)
        model_buffer.seek(0)
        model = sio.load(model_buffer, trusted=untrusted_types) 
        
        data = pd.read_csv(BytesIO(data_csv_str.encode('utf-8')))
        
        # --- VALIDATION GUARDRAIL ---
        # Run lightweight checks before any heavy processing.
        # Raises InputValidationError (which skips Celery retries) on failure.
        self.update_state(state='PROGRESS', meta={'message': 'Validating inputs...'})
        validate_audit_inputs(model, data, config)
        
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

    except InputValidationError as e:
        # User-input errors: fail immediately, do NOT retry.
        elapsed_time = time.time() - start_time
        _task_metrics["failed_tasks"] += 1
        logger.warning(f"Task {task_id} failed input validation in {elapsed_time:.2f}s: {str(e)}")
        return {"status": "error", "error": str(e)}

    except ValueError as e:
        # Scikit-learn shape/type errors during prediction: fail immediately, do NOT retry.
        elapsed_time = time.time() - start_time
        _task_metrics["failed_tasks"] += 1
        logger.warning(f"Task {task_id} failed with ValueError in {elapsed_time:.2f}s: {str(e)}")
        return {
            "status": "error",
            "error": f"Model-data incompatibility: {str(e)}. "
                     f"Please ensure your dataset matches the features your model was trained on."
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

