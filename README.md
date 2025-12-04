# 🚀 Bias & Fairness Debugger

A plug-and-play Python toolkit and Streamlit dashboard to audit trained ML models for hidden biases and suggest mitigation strategies.
Check it out here! https://fairness-troops.streamlit.app/

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
git clone https://github.com/DishaGarg-16/fairness_troops.git
cd fairness_troops
```

### 2. Create a Virtual Environment

**For macOS/Linux**
```

python3 -m venv venv
source venv/bin/activate

```

**For Windows**
```

python -m venv venv
.\venv\Scripts\activate

```

### 3. Install Dependencies

```

pip install -r requirements.txt

```

### 4. Run the Streamlit Dashboard

```

streamlit run app/app.py

```

This will open the Bias & Fairness Debugger in your web browser.

---

## 🕹️ Using the Dashboard

1. **Upload Model** – Provide a trained model (`model.pkl` or `model.joblib`).  
2. **Upload Data** – Upload the test dataset (`test_data.csv`).  
3. **Configure Audit** –  
   - Select *Target Variable* (e.g., `is_approved`)  
   - Select *Protected Attribute* (e.g., `gender`, `race`, `age_group`)
4. **Run Audit** – Click **“Run Fairness Audit”**.
5. **Review Results** –  
   - View Disparate Impact & Equal Opportunity metrics  
   - Read plain-English summaries  
   - Get mitigation recommendations (e.g., Reweighting)  
   - Download a CSV with new sample weights  

---

## 📂 Project Structure

```

fairness_troops/
├── app/
│   └── app.py             \# Streamlit dashboard
├── src/
│   └── fairness_troops/   \# Core Python package
│       ├── __init__.py
│       ├── metrics.py     \# Fairness calculation functions
│       └── mitigation.py  \# Bias mitigation methods
├── tests/                 \# Pytest unit tests
│   └── test_metrics.py
├── examples/              \# Example models and datasets
│   ├── sample_model.joblib
│   └── sample_data.csv
├── .gitignore
├── pyproject.toml         \# Build and packaging config
└── requirements.txt       \# Python dependencies

```

---

## 🤝 Contributing

Contributions are always welcome!  
To contribute:

1. Fork the repository.  
2. Create a branch:  
```

git checkout -b feature/your-feature-name

```
3. Commit your changes:  
```

git commit -m "Add some feature"

```
4. Push to your branch:  
```

git push origin feature/your-feature-name

```
5. Open a Pull Request.

Ensure your code passes `pytest` and follows the existing code style conventions.

---

## ⚖️ License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.
