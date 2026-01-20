# src/bias_debugger/__init__.py

# This makes it so users can write:
# from bias_debugger import BiasAuditor
# instead of:
# from bias_debugger.core import BiasAuditor
from .mitigation import (
    get_reweighting_weights,
    apply_threshold_optimizer,
    apply_reject_option_classification,
    calculate_fairness_improvement
)
from .explainability import BiasExplainer
from .reporting import ReportGenerator
from .core import BiasAuditor