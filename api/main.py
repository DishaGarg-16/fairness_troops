from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from io import BytesIO
from fairness_troops import BiasAuditor
import logging
import json
import base64
from .database import engine, Base, get_db
from .models import AuditLog
from .cache import get_redis_client, generate_cache_key
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from contextlib import asynccontextmanager
from .schemas import AuditResponse, FairnessMetrics
from .tasks import run_audit_task
from celery.result import AsyncResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(title="Fairness Troops API", lifespan=lifespan)

# Add CORS
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

@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    health_status = {"status": "healthy", "postgres": "unknown", "redis": "unknown"}
    
    # Check Postgres
    try:
        await db.execute(text("SELECT 1"))
        health_status["postgres"] = "connected"
    except Exception as e:
        health_status["postgres"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    # Check Redis
    try:
        redis = await get_redis_client()
        await redis.ping()
        health_status["redis"] = "connected"
    except Exception as e:
        health_status["redis"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
        
    return health_status

@app.post("/audit")
async def start_audit_task(
    target_col: str = Form(...),
    sensitive_col: str = Form(...),
    privileged_group: str = Form(...),
    unprivileged_group: str = Form(...),
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
):
    try:
        logger.info(f"Received audit request. Target: {target_col}, Sensitive: {sensitive_col}")

        # Read files content
        model_content = await model_file.read()
        data_content = await data_file.read()
        
        # Encode for Celery
        model_b64 = base64.b64encode(model_content).decode('utf-8')
        # We assume data is text (CSV), but let's carefully handle encoding if bytes
        # data_content is bytes. pandas.read_csv accepts bytesio
        # But we need to pass a string to Celery ideally or b64 encoded bytes
        data_str = data_content.decode('utf-8')

        config = {
            "target_col": target_col,
            "sensitive_col": sensitive_col,
            "privileged_group": privileged_group,
            "unprivileged_group": unprivileged_group
        }
        
        # Start Celery Task
        task = run_audit_task.delay(model_b64, data_str, config)
        
        return {"task_id": task.id, "status": "processing"}

    except Exception as e:
        logger.error(f"Audit initiation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result/{task_id}")
async def get_audit_result(task_id: str):
    task_result = AsyncResult(task_id)
    
    if task_result.state == 'PENDING':
        return {"task_id": task_id, "state": "PENDING", "status": "Pending..."}
    
    elif task_result.state == 'PROGRESS':
        return {
            "task_id": task_id, 
            "state": "PROGRESS", 
            "status": task_result.info.get('message', 'Processing...')
        }
    
    elif task_result.state == 'SUCCESS':
        # Result is the dict returned by run_audit_task
        return {
            "task_id": task_id, 
            "state": "SUCCESS", 
            "result": task_result.result
        }
        
    elif task_result.state == 'FAILURE':
        return {
            "task_id": task_id, 
            "state": "FAILURE", 
            "error": str(task_result.info)
        }
        
    return {"task_id": task_id, "state": task_result.state}
```
