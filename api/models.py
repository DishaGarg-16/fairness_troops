from sqlalchemy import Column, Integer, String, JSON, DateTime, Float
from sqlalchemy.sql import func
from .database import Base

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    target_col = Column(String)
    sensitive_col = Column(String)
    privileged_group = Column(String)
    unprivileged_group = Column(String)
    metrics_result = Column(JSON) # Store the full report as JSON
    input_hash = Column(String, index=True, nullable=True) # For caching/uniqueness checks if needed
