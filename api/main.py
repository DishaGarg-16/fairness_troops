from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from io import BytesIO
from fairness_troops import BiasAuditor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fairness Troops API")

# Add CORS so frontend can talk to backend (even if on different ports locally)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Fairness Troops API"}

@app.post("/audit")
async def audit_model(
    target_col: str = Form(...),
    sensitive_col: str = Form(...),
    privileged_group: str = Form(...),
    unprivileged_group: str = Form(...),
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...)
):
    try:
        logger.info(f"Received audit request. Target: {target_col}, Sensitive: {sensitive_col}")

        # Load Model
        model_content = await model_file.read()
        model = joblib.load(BytesIO(model_content))

        # Load Data
        data_content = await data_file.read()
        try:
            data = pd.read_csv(BytesIO(data_content))
        except Exception as e:
            # Try parsing line by line if strict loading fails (rare but possible with weird CSVs)
             logger.error(f"CSV Parse Error: {e}")
             raise HTTPException(status_code=400, detail="Invalid CSV file")

        # Basic validation
        if target_col not in data.columns:
             raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found in dataset")
        if sensitive_col not in data.columns:
             raise HTTPException(status_code=400, detail=f"Sensitive column '{sensitive_col}' not found in dataset")

        # Run Audit
        auditor = BiasAuditor(
            model=model,
            dataset=data,
            target_col=target_col,
            sensitive_col=sensitive_col,
            privileged_group=privileged_group,
            unprivileged_group=unprivileged_group
        )

        report = auditor.run_audit()
        
        # We need to serialize the report for JSON
        # If the report contains numpy types, we need to convert them
        # (Current Simple implementation returns floats which are fine)
        
        # Return predictions so frontend can visualize them
        predictions = auditor.y_pred.tolist()
        
        return {
            "status": "success",
            "metrics": report,
            "predictions": predictions,
            "mitigation_weights": auditor.get_mitigation_weights().tolist()
        }

    except Exception as e:
        logger.error(f"Audit failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
