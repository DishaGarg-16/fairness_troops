from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from io import BytesIO
from fairness_troops import BiasAuditor
import logging
import json
import base64
from pydantic import ValidationError
from .database import engine, Base, get_db
from .models import AuditLog
from .cache import get_redis_client, generate_cache_key
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from contextlib import asynccontextmanager
from .schemas import (
    AuditResponse, FairnessMetrics, DatasetConfig, 
    TaskStatusResponse, TaskStartResponse, HealthResponse,
    MAX_FILE_SIZE_BYTES
)
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

@app.get("/health", response_model=HealthResponse)
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
        
    return HealthResponse(**health_status)

@app.post("/audit", response_model=TaskStartResponse)
async def start_audit_task(
    target_col: str = Form(...),
    sensitive_col: str = Form(...),
    privileged_group: str = Form(...),
    unprivileged_group: str = Form(...),
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
):
    # Validate form inputs using Pydantic DatasetConfig
    try:
        config = DatasetConfig(
            target_col=target_col,
            sensitive_col=sensitive_col,
            privileged_group=privileged_group,
            unprivileged_group=unprivileged_group
        )
    except ValidationError as e:
        # Extract user-friendly error messages
        errors = e.errors()
        error_messages = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in errors])
        raise HTTPException(status_code=422, detail=f"Validation error: {error_messages}")

    logger.info(f"Received audit request. Target: {config.target_col}, Sensitive: {config.sensitive_col}")

    # Validate Model File Extension
    if not model_file.filename or not model_file.filename.endswith('.skops'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid model file. Only .skops files are allowed for security."
        )
    
    # Validate Data File Extension
    if not data_file.filename or not data_file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Invalid data file. Only .csv files are allowed."
        )
    
    try:
        # Read files content
        model_content = await model_file.read()
        data_content = await data_file.read()
        
        # Validate file sizes
        if len(model_content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400, 
                detail=f"Model file too large (max {MAX_FILE_SIZE_BYTES // (1024*1024)}MB)."
            )
        
        if len(data_content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400, 
                detail=f"Data file too large (max {MAX_FILE_SIZE_BYTES // (1024*1024)}MB)."
            )
        
        # Encode for Celery
        model_b64 = base64.b64encode(model_content).decode('utf-8')
        # Decode and validate CSV data
        try:
            data_str = data_content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid data file encoding. File must be UTF-8 encoded CSV."
            )

        config_dict = config.model_dump()
        
        # Start Celery Task
        task = run_audit_task.delay(model_b64, data_str, config_dict)
        
        return TaskStartResponse(task_id=task.id, status="processing")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audit initiation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result/{task_id}", response_model=TaskStatusResponse)
async def get_audit_result(task_id: str):
    # Validate task_id format (Celery uses UUID format)
    if not task_id or len(task_id) < 10 or len(task_id) > 50:
        raise HTTPException(status_code=400, detail="Invalid task_id format")
    
    task_result = AsyncResult(task_id)
    
    if task_result.state == 'PENDING':
        return TaskStatusResponse(task_id=task_id, state="PENDING", status="Pending...")
    
    elif task_result.state == 'PROGRESS':
        return TaskStatusResponse(
            task_id=task_id, 
            state="PROGRESS", 
            status=task_result.info.get('message', 'Processing...')
        )
    
    elif task_result.state == 'SUCCESS':
        return TaskStatusResponse(
            task_id=task_id, 
            state="SUCCESS", 
            result=task_result.result
        )
        
    elif task_result.state == 'FAILURE':
        return TaskStatusResponse(
            task_id=task_id, 
            state="FAILURE", 
            error=str(task_result.info)
        )
        
    return TaskStatusResponse(task_id=task_id, state=task_result.state)
