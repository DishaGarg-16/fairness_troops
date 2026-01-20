"""
Benchmark: Performance at Scale

This benchmark demonstrates that the fairness audit system can process 
datasets with 100k-500k samples within 30-60 seconds using Celery workers.

Results will depend on:
- Hardware (CPU cores, memory)
- Number of Celery workers
- Model complexity

Usage:
    python benchmarks/benchmark_performance.py
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairness_troops import BiasAuditor


def generate_synthetic_data(n_samples: int, n_features: int = 20) -> pd.DataFrame:
    """Generate synthetic dataset with bias for benchmarking."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate sensitive attribute (binary: 0=unprivileged, 1=privileged)
    sensitive = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    
    # Generate target with bias - privileged group more likely to get positive outcome
    noise = np.random.randn(n_samples) * 0.5
    score = X[:, 0] * 0.3 + X[:, 1] * 0.2 + sensitive * 0.4 + noise
    target = (score > 0.3).astype(int)
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['sensitive_attr'] = sensitive
    df['target'] = target
    
    return df


def train_model(df: pd.DataFrame):
    """Train a RandomForest model on the data."""
    X = df.drop(columns=['target'])
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    return model


def run_benchmark(sample_sizes: list[int] = None):
    """
    Run performance benchmark at different data sizes.
    
    Returns timing results for each sample size.
    """
    if sample_sizes is None:
        sample_sizes = [10_000, 50_000, 100_000, 250_000, 500_000]
    
    results = []
    
    print("=" * 70)
    print("PERFORMANCE BENCHMARK: Fairness Audit at Scale")
    print("=" * 70)
    print()
    
    for n_samples in sample_sizes:
        print(f"\n{'='*50}")
        print(f"Testing with {n_samples:,} samples...")
        print(f"{'='*50}")
        
        # Generate data
        print("  [1/4] Generating synthetic data...", end=" ")
        t0 = time.time()
        df = generate_synthetic_data(n_samples)
        data_gen_time = time.time() - t0
        print(f"Done ({data_gen_time:.2f}s)")
        
        # Train model
        print("  [2/4] Training model...", end=" ")
        t0 = time.time()
        model = train_model(df)
        train_time = time.time() - t0
        print(f"Done ({train_time:.2f}s)")
        
        # Run fairness audit (this is what we're benchmarking)
        print("  [3/4] Running fairness audit...", end=" ")
        t0 = time.time()
        
        auditor = BiasAuditor(
            model=model,
            dataset=df,
            target_col='target',
            sensitive_col='sensitive_attr',
            privileged_group=1,
            unprivileged_group=0
        )
        
        # Run all metrics
        report = auditor.run_audit()
        
        # Get mitigation weights
        weights = auditor.get_mitigation_weights()
        
        audit_time = time.time() - t0
        print(f"Done ({audit_time:.2f}s)")
        
        # Generate visuals (optional, adds time)
        print("  [4/4] Generating visualizations...", end=" ")
        t0 = time.time()
        visuals = auditor.get_visuals()
        visual_time = time.time() - t0
        print(f"Done ({visual_time:.2f}s)")
        
        total_time = audit_time + visual_time
        
        result = {
            "samples": n_samples,
            "data_gen_seconds": round(data_gen_time, 2),
            "training_seconds": round(train_time, 2),
            "audit_seconds": round(audit_time, 2),
            "visualization_seconds": round(visual_time, 2),
            "total_audit_time_seconds": round(total_time, 2),
            "disparate_impact": round(report.get('disparate_impact', 0), 4),
            "spd": round(report.get('statistical_parity_diff', 0), 4),
        }
        results.append(result)
        
        print(f"\n  📊 Results for {n_samples:,} samples:")
        print(f"     • Audit time: {audit_time:.2f}s")
        print(f"     • Total (audit + visuals): {total_time:.2f}s")
        print(f"     • Disparate Impact: {report.get('disparate_impact', 0):.4f}")
        print(f"     • Status: {'✅ PASS' if total_time < 60 else '⚠️ SLOW'} (target: <60s)")
    
    # Summary
    print("\n")
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print("\n| Samples | Audit Time | Total Time | Status |")
    print("|---------|------------|------------|--------|")
    for r in results:
        status = "✅ PASS" if r["total_audit_time_seconds"] < 60 else "⚠️ SLOW"
        print(f"| {r['samples']:>7,} | {r['audit_seconds']:>8.2f}s | {r['total_audit_time_seconds']:>8.2f}s | {status} |")
    
    # Performance claim validation
    print("\n VALIDATION:")
    claim_samples = [100_000, 250_000, 500_000]
    passing = all(
        r["total_audit_time_seconds"] < 60 
        for r in results 
        if r["samples"] in claim_samples
    )
    
    if passing:
        print("✅ VALIDATED: 100k-500k samples processed in under 60 seconds")
    else:
        print("⚠️ PARTIAL: Some sample sizes exceeded 60 seconds")
        print("   Consider: scaling workers, using faster hardware, or adjusting claim")
    
    return results


if __name__ == "__main__":
    results = run_benchmark()
    
    # Save results to file
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), "performance_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n📁 Results saved to: {output_path}")
