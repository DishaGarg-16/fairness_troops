# tests/test_metrics.py
import pandas as pd
import pytest
from fairness_troops import metrics

@pytest.fixture
def sample_data_perfect_fairness():
    # 10 privileged, 5 get loan (50%)
    # 10 unprivileged, 5 get loan (50%)
    # DI should be 1.0
    y_pred = pd.Series(
        [1,1,1,1,1,0,0,0,0,0] +  # Privileged
        [1,1,1,1,1,0,0,0,0,0]    # Unprivileged
    )
    sensitive = pd.Series(
        ['priv']*10 + ['unpriv']*10
    )
    # y_true needed for EOD. Let's make it simple.
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
    di = metrics.calculate_disparate_impact(
        y_pred, sensitive, 'priv', 'unpriv'
    )
    assert di == 1.0

def test_disparate_impact_clear_bias(sample_data_clear_bias):
    _, y_pred, sensitive = sample_data_clear_bias
    di = metrics.calculate_disparate_impact(
        y_pred, sensitive, 'priv', 'unpriv'
    )
    assert di == 0.5

def test_eod_perfect_fairness(sample_data_perfect_fairness):
    y_true, y_pred, sensitive = sample_data_perfect_fairness
    eod = metrics.calculate_equal_opportunity_difference(
        y_true, y_pred, sensitive, 'priv', 'unpriv'
    )
    assert eod == 0.0

def test_eod_clear_bias(sample_data_clear_bias):
    y_true, y_pred, sensitive = sample_data_clear_bias
    eod = metrics.calculate_equal_opportunity_difference(
        y_true, y_pred, sensitive, 'priv', 'unpriv'
    )
    assert eod == pytest.approx((4/9) - (8/9)) # Use approx for float comparison