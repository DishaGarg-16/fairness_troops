# 🚀 Bias & Fairness Debugger

A plug-and-play Python toolkit and Streamlit dashboard to audit trained ML models for hidden biases and suggest mitigation strategies.

## The Problem

Companies deploying ML models for lending, hiring, or healthcare face significant legal and PR risks from biased models. This tool provides a simple "health check" for a trained model to identify and visualize these biases *before* deployment.

## Features

* **Audit Metrics**: Calculates key fairness metrics like:
    * **Disparate Impact Ratio (DI)**
    * **Equal Opportunity Difference (EOD)**
* **Executive-Ready Visuals**: Generates simple bar charts to visualize disparities in outcomes and true positive rates.
* **Actionable Mitigation**: Suggests pre-processing mitigation strategies, starting with **Reweighting**, and provides a downloadable weights file.
* **Plug-and-Play**: Works with any `scikit-learn`-compatible model (`.joblib` or `.pkl`).

## Tech Stack

* **Python Package**: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
* **Web Dashboard**: `streamlit`
* **Packaging**: `uv`, `setuptools`, `pyproject.toml`
* **Testing**: `pytest`

---

## 🚀 How to Run

### 1. Setup

**Clone the repository:**
```bash
git clone [https://github.com/yourusername/bias-fairness-debugger.git](https://github.com/yourusername/bias-fairness-debugger.git)
cd bias-fairness-debugger