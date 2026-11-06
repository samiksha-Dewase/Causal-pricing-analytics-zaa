# Treatment & Control Specification Protocol

**Author:** Member 2 — Causal Design & Directed Acyclic Graph (DAG) Lead  
**Project:** Causal Inference & Synthetic Control Engine for Algorithmic Pricing  
**Organization:** Zaalima Development — Enterprise Data Analytics  
**Document:** Deliverable — Treatment/Control Definition  

---

## 1. Overview

This document specifies the exact unit partitions, operational inclusion rules, and econometric boundaries distinguishing the **Treated Unit** from the **Donor Pool (Control Units)** across the 52-week observation panel.

---

## 2. Treated Entity Definition

* **Unit Code:** `Store_1`
* **SKU Code:** `PROD_101` (Flagship Energy Drink 500ml)
* **Pre-Intervention Baseline:** Weeks 1 through 35 ($T_0 = 35$)
  - Pricing regime: Legacy baseline cost-plus shelf price ($\approx \$3.50$).
  - Mean weekly sales: $\approx 540$ units.
* **Post-Intervention Window:** Weeks 36 through 52 (17 weeks)
  - Pricing regime: Algorithmic pricing experiment initiated at Week 36.
  - Price jump: **$+15.0\%$** (to $\approx \$4.02$).
  - Expected sales drop: **$\approx -22.5\%$** ($\eta = -1.50$).

---

## 3. Donor Pool (Control Units) Specification

The donor pool consists of **9 untreated retail store locations** across distinct geographic micro-markets:

$$\mathcal{J} = \{\text{Store\_2},\, \text{Store\_3},\, \text{Store\_4},\, \text{Store\_5},\, \text{Store\_6},\, \text{Store\_7},\, \text{Store\_8},\, \text{Store\_9},\, \text{Store\_10}\}$$

### Selection & SUTVA Boundary Criteria

| Criteria | Operational Rule | Verification Status |
| :--- | :--- | :--- |
| **No Pricing Intervention** | Donor stores maintained legacy pricing throughout weeks 1–52. | Verified (`treatment_active == 0` for all donor stores). |
| **Geographic Non-Interference** | Stores are non-adjacent; no cross-store cannibalization or consumer shopping arbitrage. | Protected via spatial clustering design. |
| **Complete Balanced Panel** | Continuous weekly telemetry for 52 weeks without missing observations. | Verified (52 rows per store $\times$ 10 stores = 520 rows for `PROD_101`). |
| **Covariate Overlap (Convex Hull)** | Historical footfall, marketing spend, and competitor price metrics bracket `Store_1`. | Confirmed via Member 1 EDA summary statistics. |

---

## 4. Matrix Formulation for Synthetic Control (Handoff to Member 3)

Let $Y_{1}$ be the $(T \times 1)$ vector of observed weekly sales for `Store_1`, and $\mathbf{Y}_0$ be the $(T \times 9)$ matrix of sales across the donor pool:

$$\mathbf{Y} = \begin{bmatrix} Y_{1, 1} & Y_{2, 1} & \cdots & Y_{10, 1} \\ Y_{1, 2} & Y_{2, 2} & \cdots & Y_{10, 2} \\ \vdots & \vdots & \ddots & \vdots \\ Y_{1, 35} & Y_{2, 35} & \cdots & Y_{10, 35} \\ \hline Y_{1, 36} & Y_{2, 36} & \cdots & Y_{10, 36} \\ \vdots & \vdots & \ddots & \vdots \\ Y_{1, 52} & Y_{2, 52} & \cdots & Y_{10, 52} \end{bmatrix} = \begin{bmatrix} \mathbf{Y}_{1, pre} & \mathbf{Y}_{0, pre} \\ \hline \mathbf{Y}_{1, post} & \mathbf{Y}_{0, post} \end{bmatrix}$$

Member 3 will optimize the weight vector $\mathbf{W}^* = (w_2, \dots, w_{10})^T$ subject to:
$$\min_{\mathbf{W}} \|\mathbf{Y}_{1, pre} - \mathbf{Y}_{0, pre} \mathbf{W}\|_2^2 \quad \text{s.t.} \quad \sum_{j=2}^{10} w_j = 1, \quad w_j \ge 0$$
