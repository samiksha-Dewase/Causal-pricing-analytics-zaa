# Causal Inference & Synthetic Control Engine for Algorithmic Pricing
## Enterprise-Level Data Analytics Project — Member 2: Causal Design & DAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Causal Framework](https://img.shields.io/badge/Causal%20Framework-DoWhy%20%7C%20NetworkX-emerald.svg)](https://github.com/py-why/dowhy)
[![Methodology](https://img.shields.io/badge/Methodology-Causal%20DAG%20%26%20Backdoor%20Identification-orange.svg)]()

This repository implements the **Causal Design & Directed Acyclic Graph (DAG) Engine** for an enterprise-level algorithmic pricing system. In multi-store retail environments, standard A/B testing breaks down due to spatial demand substitution and market confounders. This component establishes the econometric foundation, potential outcomes framework, backdoor adjustment sets, and structural DAG required for Synthetic Control Methods (SCM) and Double Machine Learning (DML).

---

## 👥 Project & Team Context

* **Organization:** Zaalima Development — Enterprise Data Analytics
* **Role:** **Member 2 — Causal Design & Directed Acyclic Graph (DAG) Lead**
* **Upstream Ingestion:** Ingests weekly store-product panel data (`data/processed/store_product_week_panel.csv`) prepared by Member 1 (Data Engineering).
* **Downstream Handoff:** Provides identified adjustment sets, donor pool boundaries, and estimand definitions to Member 3 (Synthetic Control) and Member 4 (Double Machine Learning).

---

## 📌 Member 2 Deliverables Summary

| Deliverable | Description | File Location |
| :--- | :--- | :--- |
| **1. Causal Research Design** | Formal potential outcomes framing, potential estimands (ATT & elasticity $\eta$), mathematical identification assumptions (Unconfoundedness, SUTVA boundaries, Positivity, No Anticipation). | `docs/causal_research_design.md` |
| **2. Treatment & Control Specs** | Formal unit boundaries: **Treated Unit** (`Store_1` on `PROD_101`) vs **Donor Pool** (`Store_2`–`Store_10`), pre-treatment baseline (Weeks 1–35) vs post-treatment intervention (Weeks 36–52). | `docs/treatment_control_specification.md` |
| **3. Confounder Matrix & Bad Controls** | Confounder taxonomy: competitor telemetry, ad spend, footfall, holiday shocks. Detailed analysis of **Bad Controls** (why downstream mediators and colliders must be excluded). | `docs/confounder_documentation.md` |
| **4. Structural Causal DAG** | NetworkX structural causal graph module with automated DOT generation for DoWhy and publication-grade visual plot. | `src/causal_design/dag_builder.py` |
| **5. DoWhy Identification Engine** | Microsoft DoWhy pipeline implementing Pearl's Backdoor Criterion, initial elasticity estimation, and 3 sensitivity refuters. | `src/causal_design/dowhy_identifier.py` |
| **6. Visual DAG & Diagnostics** | Publication-grade plots showing the color-coded DAG structure and refutation sensitivity diagnostics. | `reports/causal_figures/05_causal_dag_structure.png`<br>`reports/causal_figures/06_refutation_diagnostics.png` |
| **7. Interactive Walkthrough Notebook** | Step-by-step Jupyter notebook covering data ingestion, treatment split, DAG plotting, and DoWhy refutation checks. | `notebooks/02_member2_causal_design_and_dag.ipynb` |
| **8. Master Pipeline Runner** | CLI master runner script that verifies the full Member 2 pipeline in under 3 seconds. | `run_member2_causal_design.py` |

---

## 🔬 Econometric Audit & Refutation Results

When running `python run_member2_causal_design.py`:
* **Minimal Backdoor Adjustment Set Identified:** `['avg_store_footfall', 'holiday_days', 'log_competitor_price', 'total_marketing_spend']`
* **Estimated Own-Price Elasticity ($\eta$):** **`-1.4091`** (against ground-truth target **`-1.5000`**).
* **Random Common Cause Refuter:** Passed (estimate remained unchanged at `-1.4100`).
* **Data Subset Refuter (85%):** Passed (estimate remained stable at `-1.4164`).
* **Placebo Treatment Refuter:** Passed (estimate collapsed to `+0.0772` $\approx 0$).

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Member 2 Causal Pipeline
```bash
python run_member2_causal_design.py
```

### 3. Open Interactive Jupyter Walkthrough
```bash
jupyter notebook notebooks/02_member2_causal_design_and_dag.ipynb
```
