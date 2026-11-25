# Member 4: Double Machine Learning (DML) & Interactive Dashboard Report

**Contributor:** Riyaz M (Member 4 — Double ML & Dashboard Lead)  
**Project:** Causal Inference & Synthetic Control Engine for Algorithmic Pricing (Zaalima Development)

---

## 1. Econometric Methodology: Partialling-Out Double Machine Learning

To estimate the causal own-price elasticity ($\theta$) free of confounding bias from store traffic, competitor pricing, promotional ad spend, retail holidays, and SKU-level baseline differences, we implement **Chernozhukov et al. (2018) Partialling-Out Double Machine Learning** with **5-fold cross-fitting**:

1. **First-Stage Nuisance Estimation:**
   * Outcome model: $\hat{g}(X) = \mathbb{E}[\ln(\text{units}) \mid X]$
   * Treatment model: $\hat{m}(X) = \mathbb{E}[\ln(\text{price}) \mid X]$
   * Confounder vector $X$: `log_competitor_price`, `total_marketing_spend`, `avg_store_footfall`, `holiday_days`, plus product SKU fixed effects.
2. **Orthogonalization (Out-of-Fold Residuals):**
   * $\tilde{Y}_i = Y_i - \hat{g}^{(-k(i))}(X_i)$
   * $\tilde{T}_i = T_i - \hat{m}^{(-k(i))}(X_i)$
3. **Second-Stage Causal Estimation:**
   * Regress $\tilde{Y}_i$ on $\tilde{T}_i$ with heteroskedasticity-robust (HC3) standard errors to obtain $\hat{\theta}$.

---

## 2. Benchmark Results Across Estimators

| Estimator | Estimated Elasticity ($\hat{\theta}$) | Robust Std. Error | 95% Confidence Interval | Ground Truth ($\eta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Naive OLS (Unadjusted)** | `-0.0124` | `0.0185` | `[-0.0487, 0.0239]` | `-1.5000` |
| **Member 2 DoWhy (Backdoor)** | `-1.4091` | `0.0850` | `[-1.5757, -1.2425]` | `-1.5000` |
| **Double ML (Random Forest)** | **`-1.2595`** | `0.0978` | `[-1.4512, -1.0678]` | `-1.5000` |
| **Double ML (Gradient Boosting)** | `-1.1415` | `0.0937` | `[-1.3250, -0.9579]` | `-1.5000` |
| **Double ML (Ridge Linear)** | **`-1.3268`** | `0.0902` | `[-1.5035, -1.1500]` | `-1.5000` |

### Key Takeaways
* **Elimination of Omitted-Variable Bias:** Naive OLS fails completely (`-0.0124`, $p = 0.503$) because cross-SKU price differences and store footfall confound raw price-quantity correlations. Cross-fitted DML recovers statistically significant negative price elasticity ($p < 10^{-6}$) aligned with Member 2's DoWhy benchmark (`-1.4091`) and ground truth (`-1.5000`).
* **Product-Level CATE (`PROD_101`):** On the treated SKU (`PROD_101`, Flagship Energy Drink 500ml), DML estimates $\hat{\theta}_{\text{PROD\_101}} = -1.3423$ ($95\%\text{ CI}: [-1.5705, -1.1141]$), enclosing the exact `-1.50` ground truth.
* **Post-Intervention Business Impact (`Store_1`, Weeks 36–52):** The **+14.99%** algorithmic price hike caused an average **-16.13%** drop in weekly unit volume (**-1,564.6 units** cumulative) and a net revenue change of **-$1,208.46**, demonstrating that raising price on an elastic product ($|\theta| > 1$) erodes total gross revenue.
