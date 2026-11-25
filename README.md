# Causal Inference & Synthetic Control Engine for Algorithmic Pricing
## Enterprise-Level Data Analytics Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Causal Framework](https://img.shields.io/badge/Causal%20Framework-DoWhy%20%7C%20NetworkX-emerald.svg)](https://github.com/py-why/dowhy)
[![Methodology](https://img.shields.io/badge/Methodology-Causal%20DAG%20%26%20Synthetic%20Control-orange.svg)]()

This repository implements an end-to-end **Causal Inference & Synthetic Control Engine** for an enterprise-level algorithmic pricing system. In multi-store retail environments, standard A/B testing breaks down due to spatial demand substitution and market confounders. This system combines data engineering, structural DAG identification, Synthetic Control Methods (SCM), and Double Machine Learning (DML) to estimate true causal price elasticity and counterfactual lift.

---

## 👥 Team Structure & Module Division

* **Organization:** Zaalima Development — Enterprise Data Analytics
* **Member 1 (Data Engineering Lead):** Samiksha Dewase — Raw ingestion, data cleaning, Store × Product × Week panel construction, data quality reports, EDA figures, and team Excel workbook.
* **Member 2 (Causal Design & DAG Lead):** Srilekha — Causal research design, potential outcomes framing, NetworkX structural DAG, DoWhy backdoor identification, and sensitivity refutations.
* **Member 3 (Synthetic Control Lead):** Convex donor weight optimization, counterfactual trajectory reconstruction, and in-time/space placebo audits.
* **Member 4 (Double ML & Dashboard Lead):** Cross-fitted nuisance models, Neyman-orthogonal elasticity estimation, and executive Streamlit UI.

---

## 📊 Member 1 Deliverables Summary (Data Engineering)

| Deliverable | Description | File Location |
| :--- | :--- | :--- |
| **1. Raw Transaction Ingestion** | 52-week transaction dataset across 10 retail locations, including real-world noise, competitor shocks, ad spend, and holiday waves. | `data/raw/retail_transactions_raw.csv` |
| **2. Cleaning & QA Pipeline** | Deduplication, filtering negative returns/prices, and median imputation of missing confounder values. | `src/data_pipeline/clean_dataset.py`<br>`reports/data_quality_report.md` |
| **3. Cleaned Transactions** | 10,866 verified transaction records with 100% schema integrity. | `data/processed/retail_transactions_cleaned.csv` |
| **4. Balanced Longitudinal Panel** | 1,560 rows (10 stores × 3 products × 52 weeks) with causal indicators (`is_treated_store`, `is_post_treatment`, `treatment_active`). | `data/processed/store_product_week_panel.csv` |
| **5. Master Excel Workbook** | Multi-tab Excel file with freeze-panes, currency formats, conditional formatting, and pre-built SCM matrix. | `data/processed/store_product_week_panel.xlsx` |
| **6. Visual EDA Figures** | 4 publication-quality charts validating the Week 36 intervention shock and pre-treatment parallel trends. | `reports/eda_figures/` |
| **7. Interactive Notebook** | Jupyter walkthrough of ingestion, cleaning, panel build, and EDA. | `notebooks/01_member1_data_engineering.ipynb` |
| **8. Master Pipeline Runner** | CLI master runner script verifying the entire Member 1 data pipeline. | `run_member1_pipeline.py` |

---

## 🎯 Member 2 Deliverables Summary (Causal Design & DAG)

| Deliverable | Description | File Location |
| :--- | :--- | :--- |
| **1. Causal Research Design** | Formal potential outcomes framing, estimands (ATT & elasticity $\eta$), mathematical identification assumptions. | `docs/causal_research_design.md` |
| **2. Treatment & Control Specs** | Formal unit boundaries: **Treated Unit** (`Store_1` on `PROD_101`) vs **Donor Pool** (`Store_2`–`Store_10`), pre-treatment baseline vs post-treatment intervention. | `docs/treatment_control_specification.md` |
| **3. Confounder Matrix & Bad Controls** | Confounder taxonomy and detailed analysis of **Bad Controls** (mediators and colliders). | `docs/confounder_documentation.md` |
| **4. Structural Causal DAG** | NetworkX structural causal graph module with automated DOT generation for DoWhy and visual plot. | `src/causal_design/dag_builder.py` |
| **5. DoWhy Identification Engine** | Microsoft DoWhy pipeline implementing Pearl's Backdoor Criterion, elasticity estimation, and 3 sensitivity refuters. | `src/causal_design/dowhy_identifier.py` |
| **6. Visual DAG & Diagnostics** | Publication-grade plots showing the color-coded DAG structure and refutation sensitivity diagnostics. | `reports/causal_figures/05_causal_dag_structure.png`<br>`reports/causal_figures/06_refutation_diagnostics.png` |
| **7. Interactive Notebook** | Jupyter notebook covering data ingestion, treatment split, DAG plotting, and DoWhy refutation checks. | `notebooks/02_member2_causal_design_and_dag.ipynb` |
| **8. Master Pipeline Runner** | CLI master runner script verifying the full Member 2 pipeline. | `run_member2_causal_design.py` |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Member 1 Pipeline (Data Engineering)
```bash
python run_member1_pipeline.py
```

### 3. Execute Member 2 Pipeline (Causal Design & DAG)
```bash
python run_member2_causal_design.py
```

### 4. Open Interactive Notebooks
```bash
jupyter notebook notebooks/01_member1_data_engineering.ipynb
jupyter notebook notebooks/02_member2_causal_design_and_dag.ipynb
```
