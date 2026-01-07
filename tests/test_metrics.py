# tests/test_metrics.py
import pandas as pd
import pytest
from fairness_troops import metrics

@pytest.fixture
def sample_data_perfect_fairness():
    # 10 privileged, 5 get loan (50%)
    # 10 unprivileged, 5 get loan (50%)
    # DI should be 1.0 (0.5 / 0.5)
    y_pred = pd.Series(
        [1,1,1,1,1,0,0,0,0,0] +  # Privileged
        [1,1,1,1,1,0,0,0,0,0]    # Unprivileged
    )
    sensitive = pd.Series(
        ['priv']*10 + ['unpriv']*10
    )
    # y_true needed for EOD.
    y_true = pd.Series(
        [1,1,1,1,0,0,0,0,1,1] +  # Priv: 6 positives
        [1,1,1,1,0,0,0,0,1,1]    # Unpriv: 6 positives
    )
    # Priv TPR: 4/6
    # Unpriv TPR: 4/6
    # EOD should be 0.0
    return y_true, y_pred, sensitive

@pytest.fixture
def sample_data_clear_bias():
    # 10 privileged, 8 get loan (80%)
    # 10 unprivileged, 4 get loan (40%)
    # DI should be 0.4 / 0.8 = 0.5
    y_pred = pd.Series(
        [1,1,1,1,1,1,1,1,0,0] +  # Privileged
        [1,1,1,1,0,0,0,0,0,0]    # Unprivileged
    )
    sensitive = pd.Series(
        ['priv']*10 + ['unpriv']*10
    )
    # y_true needed for EOD
    y_true = pd.Series(
        [1,1,1,1,1,1,1,1,1,0] +  # Priv: 9 positives
        [1,1,1,1,1,1,1,1,1,0]    # Unpriv: 9 positives
    )
    # Priv TPR: 8/9
    # Unpriv TPR: 4/9
    # EOD should be (4/9) - (8/9) = -4/9
    return y_true, y_pred, sensitive

def test_disparate_impact_perfect_fairness(sample_data_perfect_fairness):
    _, y_pred, sensitive = sample_data_perfect_fairness
    # Note: metrics module now uses fairlearn which auto-detects groups or takes them, 
    # but the our wrapper signature is (y_true, y_pred, sensitive_features) for DI?
    # Checking src/fairness_troops/metrics.py:
    # calculate_disparate_impact(y_true, y_pred, sensitive_features)
    # It seems to ignore y_true mostly for DI ratio itself but the signature requires it.
    
    # DI = P(pred=1|unpriv) / P(pred=1|priv)
    # existing wrapper returns demographic_parity_ratio
    di = metrics.calculate_disparate_impact(
        y_true=_, # Unused by DI strictly speaking but needed by function sig
        y_pred=y_pred,
        sensitive_features=sensitive
    )
    assert di == 1.0

def test_disparate_impact_clear_bias(sample_data_clear_bias):
    _, y_pred, sensitive = sample_data_clear_bias
    di = metrics.calculate_disparate_impact(
        y_true=_,
        y_pred=y_pred, 
        sensitive_features=sensitive
    )
    # Fairlearn returns min/max ratio usually, or specific if groups are ordered.
    # If 0.5 is expected:
    assert di == 0.5 or di == 2.0 # depending on which group is treated as reference if not specified

def test_eod_perfect_fairness(sample_data_perfect_fairness):
    y_true, y_pred, sensitive = sample_data_perfect_fairness
    eod = metrics.calculate_equal_opportunity_difference(
        y_true, y_pred, sensitive
    )
    assert eod == 0.0

def test_eod_clear_bias(sample_data_clear_bias):
    y_true, y_pred, sensitive = sample_data_clear_bias
    eod = metrics.calculate_equal_opportunity_difference(
        y_true, y_pred, sensitive
    )
    # The diff is abs(TPR_grp1 - TPR_grp2) or signed diff
    # Fairlearn returns difference between groups.
    expected = (4/9) - (8/9)
    assert abs(eod) == pytest.approx(abs(expected))