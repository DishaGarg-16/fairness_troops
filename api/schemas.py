from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional, Union, Annotated
import re

# Constants
MAX_COLUMN_NAME_LENGTH = 64
MAX_GROUP_NAME_LENGTH = 128
COLUMN_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


class FairnessMetrics(BaseModel):
    """Validated fairness metrics with reasonable bounds."""
    disparate_impact: Annotated[float, Field(ge=0.0, le=10.0, description="Disparate Impact Ratio (0-10)")]
    equal_opportunity_diff: Annotated[float, Field(ge=-1.0, le=1.0, description="Equal Opportunity Difference (-1 to 1)")]
    avg_abs_odds_diff: Annotated[float, Field(ge=0.0, le=1.0, description="Average Absolute Odds Difference (0-1)")]
    theil_index: Annotated[float, Field(ge=0.0, description="Theil Index (>=0)")]
    statistical_parity_diff: Annotated[float, Field(ge=-1.0, le=1.0, description="Statistical Parity Difference (-1 to 1)")]
    false_positive_rate_diff: Annotated[float, Field(ge=-1.0, le=1.0, description="FPR Difference (-1 to 1)")]
    false_negative_rate_diff: Annotated[float, Field(ge=-1.0, le=1.0, description="FNR Difference (-1 to 1)")]


class DatasetConfig(BaseModel):
    """Configuration for audit with validated column and group names."""
    target_col: Annotated[str, Field(
        min_length=1, 
        max_length=MAX_COLUMN_NAME_LENGTH,
        description="Target variable column name"
    )]
    sensitive_col: Annotated[str, Field(
        min_length=1, 
        max_length=MAX_COLUMN_NAME_LENGTH,
        description="Sensitive attribute column name"
    )]
    privileged_group: Annotated[str, Field(
        min_length=1, 
        max_length=MAX_GROUP_NAME_LENGTH,
        description="Privileged group value"
    )]
    unprivileged_group: Annotated[str, Field(
        min_length=1, 
        max_length=MAX_GROUP_NAME_LENGTH,
        description="Unprivileged group value"
    )]

    @field_validator('target_col', 'sensitive_col')
    @classmethod
    def validate_column_name(cls, v: str) -> str:
        """Validate column names to prevent injection and ensure valid identifiers."""
        v = v.strip()
        if not v:
            raise ValueError("Column name cannot be empty or whitespace only")
        if not COLUMN_NAME_PATTERN.match(v):
            raise ValueError(
                f"Invalid column name '{v}'. Must start with a letter or underscore, "
                "and contain only letters, numbers, and underscores."
            )
        return v

    @field_validator('privileged_group', 'unprivileged_group')
    @classmethod
    def validate_group_name(cls, v: str) -> str:
        """Validate and sanitize group names."""
        v = v.strip()
        if not v:
            raise ValueError("Group name cannot be empty or whitespace only")
        # Allow more characters for group values but prevent dangerous ones
        forbidden_chars = ['<', '>', '"', "'", ';', '\\', '\x00']
        for char in forbidden_chars:
            if char in v:
                raise ValueError(f"Group name contains forbidden character: '{char}'")
        return v

    @model_validator(mode='after')
    def validate_columns_different(self) -> 'DatasetConfig':
        """Ensure target and sensitive columns are different."""
        if self.target_col == self.sensitive_col:
            raise ValueError("Target column and sensitive column must be different")
        return self

    @model_validator(mode='after')
    def validate_groups_different(self) -> 'DatasetConfig':
        """Ensure privileged and unprivileged groups are different."""
        if self.privileged_group == self.unprivileged_group:
            raise ValueError("Privileged and unprivileged groups must be different")
        return self


class AuditRequest(BaseModel):
    """Request schema for audit endpoint (used for JSON body alternative)."""
    config: DatasetConfig
    model_data_b64: Annotated[str, Field(
        min_length=1,
        description="Base64 encoded model file"
    )]
    csv_data: Annotated[str, Field(
        min_length=1,
        max_length=100 * 1024 * 1024,  # 100MB text limit
        description="CSV data as string"
    )]


class AuditResponse(BaseModel):
    """Response schema for successful audit."""
    status: Annotated[str, Field(description="Status of the audit")]
    metrics: FairnessMetrics
    predictions: List[float]
    mitigation_weights: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "metrics": {
                    "disparate_impact": 0.85,
                    "equal_opportunity_diff": -0.12,
                    "avg_abs_odds_diff": 0.08,
                    "theil_index": 0.05,
                    "statistical_parity_diff": -0.15,
                    "false_positive_rate_diff": 0.03,
                    "false_negative_rate_diff": -0.12
                },
                "predictions": [0, 1, 1, 0],
                "mitigation_weights": [1.2, 0.8, 1.0, 1.1]
            }
        }


class TaskStatusResponse(BaseModel):
    """Response schema for task status endpoint."""
    task_id: Annotated[str, Field(min_length=1, description="Celery task ID")]
    state: Annotated[str, Field(description="Current task state")]
    status: Optional[str] = Field(default=None, description="Human-readable status message")
    result: Optional[Dict] = Field(default=None, description="Task result if completed")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class TaskStartResponse(BaseModel):
    """Response schema for audit task initiation."""
    task_id: Annotated[str, Field(min_length=1, description="Celery task ID")]
    status: Annotated[str, Field(default="processing", description="Initial status")]


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: Annotated[str, Field(description="Overall health status")]
    postgres: Annotated[str, Field(description="PostgreSQL connection status")]
    redis: Annotated[str, Field(description="Redis connection status")]


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    detail: Annotated[str, Field(description="Error details")]
    error_code: Optional[str] = Field(default=None, description="Machine-readable error code")
