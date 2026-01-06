from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from io import BytesIO
from fairness_troops import BiasAuditor
import logging
import json
from .database import engine, Base, get_db
from .models import AuditLog
from .cache import get_redis_client, generate_cache_key
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from contextlib import asynccontextmanager
from .schemas import AuditResponse

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

@app.post("/audit", response_model=AuditResponse)
async def audit_model(
    target_col: str = Form(...),
    sensitive_col: str = Form(...),
    privileged_group: str = Form(...),
    unprivileged_group: str = Form(...),
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    try:
        logger.info(f"Received audit request. Target: {target_col}, Sensitive: {sensitive_col}")

        # Read files content once
        model_content = await model_file.read()
        data_content = await data_file.read()

        # Generate Cache Key
        config = {
            "target": target_col,
            "sensitive": sensitive_col,
            "priv": privileged_group,
            "unpriv": unprivileged_group
        }
        cache_key = generate_cache_key(model_content, data_content, config)

        # Check Redis Cache
        redis = await get_redis_client()
        cached_result = await redis.get(cache_key)
        if cached_result:
            logger.info("Cache HIT")
            return json.loads(cached_result)
        
        logger.info("Cache MISS - Running computation")

        # Load Model & Data
        model = joblib.load(BytesIO(model_content))
        try:
            data = pd.read_csv(BytesIO(data_content))
        except Exception as e:
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
        predictions = auditor.y_pred.tolist()
        mitigation_weights = auditor.get_mitigation_weights().tolist()
        
        result_payload = {
            "status": "success",
            "metrics": report,
            "predictions": predictions,
            "mitigation_weights": mitigation_weights
        }

        # Save to Redis (expire in 1 hour)
        await redis.set(cache_key, json.dumps(result_payload), ex=3600)

        # Save to Database
        audit_log = AuditLog(
            target_col=target_col,
            sensitive_col=sensitive_col,
            privileged_group=privileged_group,
            unprivileged_group=unprivileged_group,
            metrics_result=report,
            input_hash=cache_key
        )
        db.add(audit_log)
        await db.commit()

        return result_payload

    except Exception as e:
        logger.error(f"Audit failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
