from pydantic import BaseModel
from typing import List, Dict, Optional, Union

class FairnessMetrics(BaseModel):
    disparate_impact: float
    equal_opportunity_diff: float
    avg_abs_odds_diff: float
    theil_index: float
    statistical_parity_diff: float
    false_positive_rate_diff: float
    false_negative_rate_diff: float

class DatasetConfig(BaseModel):
    target_col: str
    sensitive_col: str
    privileged_group: str
    unprivileged_group: str

class AuditResponse(BaseModel):
    status: str
    metrics: FairnessMetrics
    predictions: List[float]
    mitigation_weights: List[float]
