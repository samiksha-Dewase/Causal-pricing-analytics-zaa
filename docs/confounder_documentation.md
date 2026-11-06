# Confounder Matrix & Causal Identification Taxonomy

**Author:** Member 2 — Causal Design & Directed Acyclic Graph (DAG) Lead  
**Project:** Causal Inference & Synthetic Control Engine for Algorithmic Pricing  
**Organization:** Zaalima Development — Enterprise Data Analytics  
**Document:** Deliverable — Confounder Documentation  

---

## 1. Executive Overview

In observational retail telemetry, pricing is not set randomly. Uncontrolled comparisons between high-price and low-price periods produce biased estimates of price elasticity ($\eta$) due to simultaneous confounders (e.g. stores raise prices during peak holiday footfall or when competitors raise prices).

To achieve formal non-parametric identification under Pearl's **Backdoor Criterion**, this document categorizes every column in `store_product_week_panel.csv` into:
1. **Confounders** (require adjustment)
2. **Mediators** (intermediate causal mechanisms — **do not condition**)
3. **Colliders** (spurious link creators — **strictly do not condition**)

---

## 2. Comprehensive Confounder Matrix

| Variable | Column in Dataset | Causal Role | Causal Mechanism | Adjustment Status |
| :--- | :--- | :--- | :--- | :--- |
| **Competitor Pricing** | `avg_competitor_price`<br>`log_competitor_price` | **Primary Confounder** | When competitors raise prices, our store feels less competitive pressure (raises prices). Simultaneously, consumers substitute competitor goods with ours (increases our sales). | **Must Adjust ($\in \mathbf{X}^*$)** |
| **Marketing Spend** | `total_marketing_spend` | **Confounder** | Marketing campaigns often synchronize with pricing events. Marketing expands product visibility and footfall, inflating unit demand. | **Must Adjust ($\in \mathbf{X}^*$)** |
| **Store Footfall** | `avg_store_footfall` | **Confounder** | High foot traffic stores support higher baseline prices and produce larger transaction volumes. | **Must Adjust ($\in \mathbf{X}^*$)** |
| **Holiday Shocks** | `holiday_days` | **Exogenous Confounder** | Retail holidays induce both surge pricing algorithms and holiday shopping demand surges. | **Must Adjust ($\in \mathbf{X}^*$)** |
| **Store Characteristics** | `store_id` (Fixed Effects) | **Baseline Heterogeneity** | Absorbs unobserved location wealth, store footprint, and regional customer demographics. | **Conditioned via Panel FE** |
| **Gross Dollar Sales** | `total_revenue`<br>`log_revenue` | **Downstream Outcome / Mediator** | $\text{Revenue} = \text{Price} \times \text{Quantity}$. Conditioning on revenue when estimating price elasticity directly absorbs the outcome mechanism. | **DO NOT Adjust (Bad Control)** |
| **Competitor Retaliation** | Post-treatment competitor reactions | **Post-Treatment Collider** | Reacts both to our price intervention and to local market demand shocks. Conditioning opens a spurious non-causal path. | **DO NOT Adjust (Collider)** |

---

## 3. "Bad Controls" Analysis: Why Mediators & Colliders Must Be Excluded

### 3.1 Mediators (Overcontrolling Bias)
A mediator lies on the causal trajectory between Treatment ($T$) and Final Business Impact ($Y$):
$$\text{Price Intervention } (T) \longrightarrow \text{Quantity Sold } (Q) \longrightarrow \text{Gross Revenue } (Y)$$

* If Member 4 were to condition on `units_sold` while evaluating the effect of price on `revenue`, the true elasticity channel is blocked, leaving only the mechanical accounting identity ($\Delta P \times Q$).
* For demand elasticity estimation: $Y = \ln(\text{units\_sold})$ and $T = \ln(\text{price})$. Intermediary conversion rates or daily checkout speeds must **not** be adjusted for.

### 3.2 Colliders (M-Bias / Berkson's Paradox)
A collider is an endogenous variable influenced by two independent ancestral nodes:
$$\text{Price Treatment } (T) \longrightarrow \text{Stockout / Retaliation } (C) \longleftarrow \text{Unobserved Demand Shock } (U_Y)$$

* Controlling for collider $C$ creates an artificial correlation between $T$ and $U_Y$, introducing bias where none previously existed.

---

## 4. Minimal Sufficient Adjustment Set ($\mathbf{X}^*$)

Under the Backdoor Criterion, conditioning on $\mathbf{X}^*$ blocks all backdoor confounding paths:

$$\mathbf{X}^* = \left\{ \ln(\text{avg\_competitor\_price}),\, \text{total\_marketing\_spend},\, \text{avg\_store\_footfall},\, \text{holiday\_days} \right\}$$

### Identification Formula for Member 4 (Double Machine Learning):
$$\mathbb{E}[Y \mid do(T = t)] = \int \mathbb{E}[Y \mid T = t, \mathbf{X}^* = \mathbf{x}] \, d\mathbb{P}(\mathbf{X}^* = \mathbf{x})$$

Residualizing both $\ln(\text{units})$ and $\ln(\text{price})$ on $\mathbf{X}^*$ isolates the unconfounded variation required to recover the true elasticity $\eta^* = -1.50$.
