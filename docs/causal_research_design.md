# Causal Research Design: Algorithmic Pricing Engine

**Author:** Member 2 — Causal Design & Directed Acyclic Graph (DAG) Lead  
**Project:** Causal Inference & Synthetic Control Engine for Algorithmic Pricing  
**Organization:** Zaalima Development — Enterprise Data Analytics  
**Version:** 1.0  
**Status:** Approved & Validated against Member 1 Handover Specification  

---

## 1. Executive Summary & Problem Formulation

Standard randomized controlled trials (A/B testing) fail in interconnected retail and digital marketplaces because individual units (stores, regions, products) violate the **Stable Unit Treatment Value Assumption (SUTVA)** due to:
1. **Spatial & Network Spillovers:** Changing price in `Store_1` can induce consumers to travel or substitute purchases across nearby stores.
2. **Competitor & Macro Confounding:** Fluctuations in competitor pricing telemetry, localized marketing spend, footfall shocks, and calendar holidays impact both the pricing algorithm and sales volume simultaneously.

To isolate the authentic causal impact of dynamic pricing changes, this project builds an econometric causal inference engine combining **Synthetic Control Methods (SCM)** and **Double Machine Learning (DML)**. This document specifies the causal research design, potential outcomes framework, treatment/control definitions, temporal cutoffs, and identification assumptions.

---

## 2. Core Causal Question & Target Estimands

### Primary Causal Question
> *"What is the true causal effect of a dynamic +15% algorithmic price increase on unit sales volume and gross revenue for flagship SKU `PROD_101` in `Store_1`, controlling for competitor price movements, ad spend, footfall, and holiday shocks?"*

### Target Causal Estimands
1. **Average Treatment Effect on the Treated (ATT via SCM):**
   $$\tau_{ATT}(t) = Y_{1, t}(1) - Y_{1, t}(0) \quad \text{for } t \ge 36$$
   Where $Y_{1, t}(1)$ is the realized sales volume of `Store_1`, and $Y_{1, t}(0) = \sum_{j=2}^{10} w_j^* Y_{j, t}$ is the counterfactual trajectory synthesized from the donor pool.

2. **Own-Price Elasticity of Demand ($\eta$ via DML):**
   $$\ln(Q_{s,t}) = \eta \cdot \ln(P_{s,t}) + g_0(\mathbf{X}_{s,t}) + \varepsilon_{s,t}$$
   $$\ln(P_{s,t}) = m_0(\mathbf{X}_{s,t}) + v_{s,t}$$
   Where $\mathbf{X}_{s,t} = \{\ln(\text{competitor\_price}),\, \text{marketing\_spend},\, \text{store\_footfall},\, \text{holiday\_days}\}$.  
   **Target Ground Truth Benchmark:** $\eta^* = -1.50$ (as specified in Member 1 handover).

---

## 3. Formal Treatment & Control Group Specification

### 3.1 Treatment Definition ($T$)
The treatment intervention is the activation of the **Algorithmic Pricing Experiment** on SKU `PROD_101` at `Store_1`:
* **Continuous Treatment:** Weekly average shelf price ($\text{avg\_unit\_price}$ or $\ln(\text{avg\_unit\_price})$), jumping from baseline $\$3.50$ to $\$4.02$ (+15.0%).
* **Binary Treatment Indicator:**
  $$D_{s,p,t} = \begin{cases} 1 & \text{if } s = \text{Store\_1}, \, p = \text{PROD\_101}, \text{ and } t \ge 36 \\ 0 & \text{otherwise} \end{cases}$$

### 3.2 Target Treated Unit
* **Store Code:** `Store_1` (Flagship metropolitan retail store).
* **Target Product:** `PROD_101` (Flagship Energy Drink 500ml).

### 3.3 Donor Pool (Control Units)
The donor pool comprises 9 unshocked retail stores operating under standard pricing:
$$\mathcal{J} = \{\text{Store\_2},\, \text{Store\_3},\, \text{Store\_4},\, \text{Store\_5},\, \text{Store\_6},\, \text{Store\_7},\, \text{Store\_8},\, \text{Store\_9},\, \text{Store\_10}\}$$

**Donor Selection & SUTVA Protection Criteria:**
1. **Absence of Direct Price Shocks:** None of the donor stores experienced algorithmic pricing shifts during weeks 1–52.
2. **Geographic Insulation:** Stores are geographically separated to prevent cross-store shopping arbitrage and supply cannibalization.
3. **Convex Hull Coverage:** Pre-treatment covariates of `Store_1` lie within the convex hull of the 9 donor stores.

---

## 4. Temporal Partitioning: Pre- vs. Post-Treatment Horizons

The dataset spans 52 calendar weeks partitioned at week 36:

| Period | Week Range | Duration | Purpose |
| :--- | :--- | :--- | :--- |
| **Pre-Treatment Baseline** | Weeks `1` to `35` | 35 Weeks | Establish parallel trajectories; optimize synthetic weights $\mathbf{W}^*$ to minimize pre-treatment MSPE. |
| **Post-Treatment Intervention** | Weeks `36` to `52` | 17 Weeks | Measure causal treatment effect $\hat{\tau}_t = Y_{1,t} - \hat{Y}_{1,t}^{synth}$. |

---

## 5. Core Econometric Identification Assumptions

To ensure that the estimated effect reflects authentic causality:

### 1. Conditional Independence / Unconfoundedness (Exchangeability)
$$(Y_i(0), Y_i(1)) \perp\!\!\perp T_i \mid \mathbf{X}_i$$
*After conditioning on competitor pricing telemetry, localized marketing spend, footfall traffic, and holiday indicators, treatment assignment is as good as randomly assigned.*

### 2. Stable Unit Treatment Value Assumption (SUTVA)
$$Y_i(\mathbf{t}) = Y_i(t_i)$$
*The sales volume of `Store_1` depends only on its own price, and donor stores' outcomes are not altered by `Store_1`'s pricing intervention.*

### 3. Positivity / Common Support
$$0 < \mathbb{P}(T_i = 1 \mid \mathbf{X}_i = \mathbf{x}) < 1 \quad \forall \mathbf{x} \in \text{supp}(\mathbf{X})$$
*Stores across different price regimes share common support in competitor pricing, footfall, and marketing spend.*

### 4. No Anticipation Assumption
$$Y_{1, t}(1) = Y_{1, t}(0) \quad \forall t < 36$$
*Store managers and consumers do not alter purchasing behavior in anticipation of the Week 36 price hike.*

---

## 6. Downstream Handoff Architecture

* **To Member 3 (Synthetic Control Lead):**
  - Donor pool: `Store_2` through `Store_10`.
  - Target: `Store_1` on `PROD_101`.
  - Optimization window: Weeks 1 to 35 ($T_0 = 35$).
  - Target benchmark: Pre-intervention $\text{MSPE} \to 0$; post-intervention sales drop $\approx -22.5\%$.
* **To Member 4 (Double ML Lead):**
  - Treatment ($T$): `log_price`
  - Outcome ($Y$): `log_units`
  - Minimal Adjustment Set ($\mathbf{X}$): `log_competitor_price`, `total_marketing_spend`, `avg_store_footfall`, `holiday_days`.
  - Target benchmark: $\hat{\theta} \approx -1.50 \pm 0.08$.
