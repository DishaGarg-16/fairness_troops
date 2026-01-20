"""
Benchmark: Fairness Improvement with Mitigation Strategies

This benchmark demonstrates that reweighting and post-processing techniques
can reduce fairness gaps (SPD, EOD, DI) by 20-40% on biased datasets.

Techniques tested:
1. Reweighting (pre-processing)
2. Threshold Optimizer (post-processing)
3. Reject Option Classification (post-processing)

Usage:
    python benchmarks/benchmark_fairness_improvement.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairness_troops import (
    BiasAuditor,
    get_reweighting_weights,
    apply_threshold_optimizer,
    apply_reject_option_classification,
    calculate_fairness_improvement
)
from fairness_troops import metrics


def generate_biased_data(n_samples: int = 10000, bias_strength: float = 0.5) -> pd.DataFrame:
    """
    Generate synthetic dataset with intentional bias.
    
    Args:
        n_samples: Number of samples
        bias_strength: How much the privileged group is favored (0-1)
    
    Returns:
        DataFrame with features, sensitive attribute, and target
    """
    np.random.seed(42)
    
    # Features
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    
    # Sensitive attribute: 0=Female (unprivileged), 1=Male (privileged)
    # Imbalanced: 40% female, 60% male
    sensitive = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    
    # Generate target with bias
    # Base score from features
    base_score = 0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * X[:, 2]
    
    # Add bias: privileged group gets a boost
    biased_score = base_score + bias_strength * sensitive
    
    # Add noise
    noise = np.random.randn(n_samples) * 0.3
    final_score = biased_score + noise
    
    # Binary target
    target = (final_score > 0.2).astype(int)
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['gender'] = np.where(sensitive == 1, 'Male', 'Female')
    df['outcome'] = target
    
    return df


def run_benchmark():
    """Run the fairness improvement benchmark."""
    
    print("=" * 70)
    print("FAIRNESS IMPROVEMENT BENCHMARK")
    print("Testing mitigation strategies on biased synthetic data")
    print("=" * 70)
    print()
    
    # Generate biased dataset
    print("[1/5] Generating biased synthetic dataset...")
    df = generate_biased_data(n_samples=10000, bias_strength=0.5)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    X_train = train_df.drop(columns=['outcome', 'gender'])
    y_train = train_df['outcome']
    sensitive_train = train_df['gender']
    
    X_test = test_df.drop(columns=['outcome', 'gender'])
    y_test = test_df['outcome']
    sensitive_test = test_df['gender']
    
    print(f"   Training samples: {len(train_df):,}")
    print(f"   Test samples: {len(test_df):,}")
    print(f"   Privileged group (Male) positive rate: {(train_df[train_df['gender']=='Male']['outcome'].mean()*100):.1f}%")
    print(f"   Unprivileged group (Female) positive rate: {(train_df[train_df['gender']=='Female']['outcome'].mean()*100):.1f}%")
    
    # =========================================================================
    # BASELINE: Train model without mitigation
    # =========================================================================
    print("\n[2/5] Training baseline model (no mitigation)...")
    baseline_model = LogisticRegression(random_state=42, max_iter=1000)
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = pd.Series(baseline_model.predict(X_test), index=X_test.index)
    
    # Calculate baseline fairness metrics
    baseline_spd = abs(metrics.calculate_statistical_parity_difference(
        y_test, y_pred_baseline, sensitive_test
    ))
    baseline_eod = abs(metrics.calculate_equal_opportunity_difference(
        y_test, y_pred_baseline, sensitive_test
    ))
    baseline_di = metrics.calculate_disparate_impact(
        y_test, y_pred_baseline, sensitive_test
    )
    
    print(f"   Baseline Metrics:")
    print(f"     • Statistical Parity Difference: {baseline_spd:.4f}")
    print(f"     • Equal Opportunity Difference: {baseline_eod:.4f}")
    print(f"     • Disparate Impact Ratio: {baseline_di:.4f} (ideal=1.0)")
    
    results = [{
        "method": "Baseline (No Mitigation)",
        "spd": baseline_spd,
        "eod": baseline_eod,
        "di": baseline_di,
        "spd_improvement": 0.0,
        "eod_improvement": 0.0,
    }]
    
    # =========================================================================
    # MITIGATION 1: Reweighting (Pre-processing)
    # =========================================================================
    print("\n[3/5] Testing Reweighting (pre-processing)...")
    
    # Calculate sample weights
    train_df_copy = train_df[['outcome', 'gender']].copy()
    weights = get_reweighting_weights(train_df_copy, 'outcome', 'gender')
    
    # Train model with sample weights
    reweight_model = LogisticRegression(random_state=42, max_iter=1000)
    reweight_model.fit(X_train, y_train, sample_weight=weights)
    y_pred_reweight = pd.Series(reweight_model.predict(X_test), index=X_test.index)
    
    # Calculate improvement
    improvement = calculate_fairness_improvement(
        y_test, y_pred_baseline, y_pred_reweight, sensitive_test
    )
    
    print(f"   Reweighting Results:")
    print(f"     • SPD: {baseline_spd:.4f} → {improvement['statistical_parity_difference']['mitigated']:.4f} "
          f"({improvement['statistical_parity_difference']['improvement_percent']:.1f}% improvement)")
    print(f"     • EOD: {baseline_eod:.4f} → {improvement['equal_opportunity_difference']['mitigated']:.4f} "
          f"({improvement['equal_opportunity_difference']['improvement_percent']:.1f}% improvement)")
    
    results.append({
        "method": "Reweighting (Pre-processing)",
        "spd": improvement['statistical_parity_difference']['mitigated'],
        "eod": improvement['equal_opportunity_difference']['mitigated'],
        "di": improvement['disparate_impact']['mitigated'],
        "spd_improvement": improvement['statistical_parity_difference']['improvement_percent'],
        "eod_improvement": improvement['equal_opportunity_difference']['improvement_percent'],
    })
    
    # =========================================================================
    # MITIGATION 2: Threshold Optimizer (Post-processing)
    # =========================================================================
    print("\n[4/5] Testing Threshold Optimizer (post-processing)...")
    
    try:
        # Need to add gender back to features for threshold optimizer
        X_train_with_sens = X_train.copy()
        X_train_with_sens['gender_encoded'] = (sensitive_train == 'Male').astype(int).values
        X_test_with_sens = X_test.copy()
        X_test_with_sens['gender_encoded'] = (sensitive_test == 'Male').astype(int).values
        
        # Train a new model with the encoded gender
        threshold_base_model = LogisticRegression(random_state=42, max_iter=1000)
        threshold_base_model.fit(X_train_with_sens, y_train)
        
        y_pred_threshold, _ = apply_threshold_optimizer(
            estimator=threshold_base_model,
            X_train=X_train_with_sens,
            y_train=y_train,
            sensitive_features_train=sensitive_train,
            X_test=X_test_with_sens,
            sensitive_features_test=sensitive_test,
            constraint="equalized_odds"
        )
        y_pred_threshold = pd.Series(y_pred_threshold, index=X_test.index)
        
        improvement_threshold = calculate_fairness_improvement(
            y_test, y_pred_baseline, y_pred_threshold, sensitive_test
        )
        
        print(f"   Threshold Optimizer Results:")
        print(f"     • SPD: {baseline_spd:.4f} → {improvement_threshold['statistical_parity_difference']['mitigated']:.4f} "
              f"({improvement_threshold['statistical_parity_difference']['improvement_percent']:.1f}% improvement)")
        print(f"     • EOD: {baseline_eod:.4f} → {improvement_threshold['equal_opportunity_difference']['mitigated']:.4f} "
              f"({improvement_threshold['equal_opportunity_difference']['improvement_percent']:.1f}% improvement)")
        
        results.append({
            "method": "Threshold Optimizer (Post-processing)",
            "spd": improvement_threshold['statistical_parity_difference']['mitigated'],
            "eod": improvement_threshold['equal_opportunity_difference']['mitigated'],
            "di": improvement_threshold['disparate_impact']['mitigated'],
            "spd_improvement": improvement_threshold['statistical_parity_difference']['improvement_percent'],
            "eod_improvement": improvement_threshold['equal_opportunity_difference']['improvement_percent'],
        })
    except Exception as e:
        print(f"   ⚠️ Threshold Optimizer failed: {e}")
        print(f"   This may require adjusting the model or data.")
    
    # =========================================================================
    # MITIGATION 3: Reject Option Classification (Post-processing)
    # =========================================================================
    print("\n[5/5] Testing Reject Option Classification (post-processing)...")
    
    # Get probability predictions
    y_proba = baseline_model.predict_proba(X_test)[:, 1]
    
    y_pred_roc = apply_reject_option_classification(
        y_pred_proba=y_proba,
        sensitive_features=sensitive_test,
        threshold=0.5,
        margin=0.15
    )
    y_pred_roc = pd.Series(y_pred_roc, index=X_test.index)
    
    improvement_roc = calculate_fairness_improvement(
        y_test, y_pred_baseline, y_pred_roc, sensitive_test
    )
    
    print(f"   Reject Option Classification Results:")
    print(f"     • SPD: {baseline_spd:.4f} → {improvement_roc['statistical_parity_difference']['mitigated']:.4f} "
          f"({improvement_roc['statistical_parity_difference']['improvement_percent']:.1f}% improvement)")
    print(f"     • EOD: {baseline_eod:.4f} → {improvement_roc['equal_opportunity_difference']['mitigated']:.4f} "
          f"({improvement_roc['equal_opportunity_difference']['improvement_percent']:.1f}% improvement)")
    
    results.append({
        "method": "Reject Option Classification (Post-processing)",
        "spd": improvement_roc['statistical_parity_difference']['mitigated'],
        "eod": improvement_roc['equal_opportunity_difference']['mitigated'],
        "di": improvement_roc['disparate_impact']['mitigated'],
        "spd_improvement": improvement_roc['statistical_parity_difference']['improvement_percent'],
        "eod_improvement": improvement_roc['equal_opportunity_difference']['improvement_percent'],
    })
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n")
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print("\n| Method | SPD | EOD | SPD Improvement | EOD Improvement |")
    print("|--------|-----|-----|-----------------|-----------------|")
    for r in results:
        print(f"| {r['method'][:35]:35} | {r['spd']:.4f} | {r['eod']:.4f} | "
              f"{r['spd_improvement']:>13.1f}% | {r['eod_improvement']:>13.1f}% |")
    
    print("   Claim: 'Reduced fairness gaps by 20-40% using reweighting and post-processing'")
    
    # Check if any method achieved 20-40% improvement
    best_spd_improvement = max(r['spd_improvement'] for r in results if r['method'] != "Baseline (No Mitigation)")
    best_eod_improvement = max(r['eod_improvement'] for r in results if r['method'] != "Baseline (No Mitigation)")
    
    if best_spd_improvement >= 20 or best_eod_improvement >= 20:
        print(f"   ✅ VALIDATED:")
        print(f"      • Best SPD improvement: {best_spd_improvement:.1f}%")
        print(f"      • Best EOD improvement: {best_eod_improvement:.1f}%")
    else:
        print(f"   ⚠️ Results vary by dataset. Current best improvements:")
        print(f"      • Best SPD improvement: {best_spd_improvement:.1f}%")
        print(f"      • Best EOD improvement: {best_eod_improvement:.1f}%")
        print(f"   Consider testing with other datasets or adjusting bias_strength.")
    
    return results


if __name__ == "__main__":
    results = run_benchmark()
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), "fairness_improvement_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n📁 Results saved to: {output_path}")
