# Causal Inference & Synthetic Control Engine for Algorithmic Pricing

An end-to-end econometric and machine learning engine to evaluate the true causal impact of algorithmic pricing changes on retail demand and revenue.

---

## Team Division & Module Responsibilities

| Role | Contributor | Focus Area | Status |
| :--- | :--- | :--- | :---: |
| **Member 1** | Samiksha Dewase | Data Engineering, Balanced Store-Product-Week Panel, QA & EDA | ✅ Complete |
| **Member 2** | Srilekha | Causal Design, Structural DAG & DoWhy Backdoor Identification | ✅ Complete |
| **Member 3** | Omkar Jasud | Synthetic Control Method (SCM) & Donor Pool Weights | ✅ Complete |
| **Member 4** | **Riyaz M** | **Double Machine Learning (DML), CATE & Interactive Streamlit Dashboard** | ✅ **Complete** |

---

## Member 4 Deliverables Summary (`Riyaz M`)

| Deliverable | Description | File Path |
| :--- | :--- | :--- |
| **1. Double ML Engine** | 5-fold cross-fitted Partialling-Out Double ML (`RandomForest`, `GradientBoosting`, `Ridge`), CATE segmentation, and counterfactual lift calculator. | `src/dml/double_ml_engine.py` |
| **2. DML Diagnostic Plots** | Publication-grade benchmark forest plot, orthogonalized residual regression fit, CATE heterogeneity chart, and counterfactual demand/revenue trajectory. | `src/dml/dml_visualizations.py` & `reports/dml_figures/07..10_*.png` |
| **3. Master DML Runner** | Single-command CLI script executing the full Member 4 estimation, CATE, counterfactual, and plotting pipeline. | `run_member4_dml_pipeline.py` |
| **4. Interactive Dashboard** | 4-tab Streamlit executive dashboard with KPI cards, estimator comparison, CATE explorer, and interactive what-if price simulator. | `app.py` |
| **5. Technical Report & Outputs** | Econometric writeup, JSON summary, model comparison CSV, CATE CSV, orthogonal residuals CSV, and Store_1 counterfactual CSV. | `docs/member4_dml_report.md` & `reports/dml_results/` |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Member 1 Data Pipeline (if panel data needs regeneration)
```bash
python run_member1_pipeline.py
```

### 3. Run Member 4 Double Machine Learning Pipeline
```bash
python run_member4_dml_pipeline.py
```

### 4. Launch Interactive Executive Dashboard
```bash
streamlit run app.py
```
