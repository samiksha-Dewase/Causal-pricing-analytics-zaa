import os
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import statsmodels.api as sm


NUMERIC_CONFOUNDERS = [
    "log_competitor_price",
    "total_marketing_spend",
    "avg_store_footfall",
    "holiday_days",
]


def _build_design_matrix(df: pd.DataFrame, confounders: list = None, include_product_fe: bool = True) -> np.ndarray:
    """Build confounder matrix X with optional product fixed effects to isolate within-SKU price variation."""
    if confounders is None:
        confounders = NUMERIC_CONFOUNDERS
    X_df = df[confounders].copy()
    if include_product_fe and "product_id" in df.columns and df["product_id"].nunique() > 1:
        prod_dummies = pd.get_dummies(df["product_id"], prefix="sku", drop_first=True, dtype=float)
        X_df = pd.concat([X_df, prod_dummies], axis=1)
    return X_df.to_numpy(dtype=float)


def _cross_fit_residuals(X: np.ndarray, y: np.ndarray, t: np.ndarray, model_y, model_t, n_splits: int = 5, seed: int = 42):
    """5-fold cross-fitting to compute orthogonalized outcome and treatment residuals."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_res = np.zeros_like(y, dtype=float)
    t_res = np.zeros_like(t, dtype=float)
    y_hat = np.zeros_like(y, dtype=float)
    t_hat = np.zeros_like(t, dtype=float)

    for train_idx, val_idx in kf.split(X):
        my = clone(model_y)
        mt = clone(model_t)
        my.fit(X[train_idx], y[train_idx])
        mt.fit(X[train_idx], t[train_idx])

        y_hat[val_idx] = my.predict(X[val_idx])
        t_hat[val_idx] = mt.predict(X[val_idx])
        y_res[val_idx] = y[val_idx] - y_hat[val_idx]
        t_res[val_idx] = t[val_idx] - t_hat[val_idx]

    return y_res, t_res, y_hat, t_hat


def estimate_dml_elasticity(
    df: pd.DataFrame,
    outcome_col: str = "log_units",
    treatment_col: str = "log_price",
    confounders: list = None,
    learner: str = "random_forest",
    n_splits: int = 5,
    seed: int = 42,
    include_product_fe: bool = True,
) -> dict:
    """
    Partialling-Out Double Machine Learning (Chernozhukov et al., 2018):
      Y_tilde = Y - E[Y | X]
      T_tilde = T - E[T | X]
      theta_hat = argmin_theta sum (Y_tilde - theta * T_tilde)^2
    """
    if learner == "random_forest":
        model_y = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=-1)
        model_t = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    elif learner == "gradient_boosting":
        model_y = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=seed)
        model_t = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=seed)
    elif learner == "ridge":
        model_y = RidgeCV(alphas=np.logspace(-3, 3, 25))
        model_t = RidgeCV(alphas=np.logspace(-3, 3, 25))
    else:
        raise ValueError(f"Unsupported learner: {learner}")

    X = _build_design_matrix(df, confounders=confounders, include_product_fe=include_product_fe)
    y = df[outcome_col].to_numpy(dtype=float)
    t = df[treatment_col].to_numpy(dtype=float)

    y_res, t_res, y_hat, t_hat = _cross_fit_residuals(X, y, t, model_y, model_t, n_splits=n_splits, seed=seed)

    # Second-stage orthogonalized regression with heteroskedasticity-robust (HC3) SE
    ols_res = sm.OLS(y_res, t_res).fit(cov_type="HC3")
    theta = float(ols_res.params[0])
    se = float(ols_res.bse[0])
    ci_low, ci_high = [float(v) for v in ols_res.conf_int()[0]]
    p_value = float(ols_res.pvalues[0])

    r2_y = float(1.0 - np.var(y_res) / np.var(y))
    r2_t = float(1.0 - np.var(t_res) / np.var(t))

    return {
        "learner": learner,
        "theta_elasticity": theta,
        "std_error": se,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "p_value": p_value,
        "first_stage_r2_y": r2_y,
        "first_stage_r2_t": r2_t,
        "n_obs": len(df),
        "y_res": y_res,
        "t_res": t_res,
    }


def estimate_cate_by_segment(df: pd.DataFrame, confounders: list = None, seed: int = 42) -> pd.DataFrame:
    """Estimate Conditional Average Treatment Effects (CATE) across products, store groups, and individual stores."""
    rows = []

    # 1. By Product SKU
    for prod_id, sub in df.groupby("product_id"):
        res = estimate_dml_elasticity(sub, confounders=confounders, learner="random_forest", seed=seed, include_product_fe=False)
        rows.append({
            "segment_type": "Product",
            "segment_value": f"{prod_id} ({sub['product_name'].iloc[0]})",
            "n_obs": res["n_obs"],
            "theta_elasticity": round(res["theta_elasticity"], 4),
            "std_error": round(res["std_error"], 4),
            "ci_95_low": round(res["ci_95_low"], 4),
            "ci_95_high": round(res["ci_95_high"], 4),
            "p_value": round(res["p_value"], 6),
        })

    # 2. Treated Store vs Donor Pool
    for label, mask in [
        ("Treated Store (Store_1)", df["is_treated_store"] == 1),
        ("Donor Pool (Store_2-10)", df["is_treated_store"] == 0),
    ]:
        sub = df[mask]
        res = estimate_dml_elasticity(sub, confounders=confounders, learner="random_forest", seed=seed, include_product_fe=True)
        rows.append({
            "segment_type": "Store Group",
            "segment_value": label,
            "n_obs": res["n_obs"],
            "theta_elasticity": round(res["theta_elasticity"], 4),
            "std_error": round(res["std_error"], 4),
            "ci_95_low": round(res["ci_95_low"], 4),
            "ci_95_high": round(res["ci_95_high"], 4),
            "p_value": round(res["p_value"], 6),
        })

    # 3. Store-level CATE across all 10 stores
    for store_id, sub in df.groupby("store_id"):
        res = estimate_dml_elasticity(sub, confounders=confounders, learner="ridge", seed=seed, include_product_fe=True)
        rows.append({
            "segment_type": "Store",
            "segment_value": store_id,
            "n_obs": res["n_obs"],
            "theta_elasticity": round(res["theta_elasticity"], 4),
            "std_error": round(res["std_error"], 4),
            "ci_95_low": round(res["ci_95_low"], 4),
            "ci_95_high": round(res["ci_95_high"], 4),
            "p_value": round(res["p_value"], 6),
        })

    return pd.DataFrame(rows)


def compute_intervention_counterfactual(df: pd.DataFrame, theta_elasticity: float) -> pd.DataFrame:
    """
    Evaluate Store_1 / PROD_101 intervention (Weeks 36-52) using DML elasticity:
      Counterfactual log_units = actual log_units - theta * (log_price - baseline_log_price)
    """
    sub = df[(df["store_id"] == "Store_1") & (df["product_id"] == "PROD_101")].sort_values("week").copy()
    baseline_log_price = sub[sub["week"] <= 35]["log_price"].mean()
    baseline_price = float(np.exp(baseline_log_price))

    sub["baseline_price"] = round(baseline_price, 4)
    sub["price_delta_pct"] = np.where(
        sub["week"] >= 36,
        (sub["avg_unit_price"] - baseline_price) / baseline_price * 100.0,
        0.0,
    )
    sub["counterfactual_units"] = np.where(
        sub["week"] >= 36,
        sub["total_units_sold"] * np.exp(-theta_elasticity * (sub["log_price"] - baseline_log_price)),
        sub["total_units_sold"],
    )
    sub["causal_units_lift"] = sub["total_units_sold"] - sub["counterfactual_units"]
    sub["causal_units_lift_pct"] = (sub["causal_units_lift"] / sub["counterfactual_units"]) * 100.0
    sub["counterfactual_revenue"] = sub["counterfactual_units"] * baseline_price
    sub["causal_revenue_lift"] = sub["total_revenue"] - sub["counterfactual_revenue"]
    return sub


def run_full_dml_analysis(panel_path: str = "data/processed/store_product_week_panel.csv", out_dir: str = "reports/dml_results") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(panel_path)

    # Naive unadjusted OLS elasticity (biased baseline)
    naive_ols = sm.OLS(df["log_units"], sm.add_constant(df["log_price"])).fit(cov_type="HC3")
    naive_theta = float(naive_ols.params["log_price"])
    naive_se = float(naive_ols.bse["log_price"])

    model_rows = [
        {
            "model": "Naive OLS (Unadjusted)",
            "theta_elasticity": round(naive_theta, 4),
            "std_error": round(naive_se, 4),
            "ci_95_low": round(float(naive_ols.conf_int().loc["log_price", 0]), 4),
            "ci_95_high": round(float(naive_ols.conf_int().loc["log_price", 1]), 4),
            "p_value": round(float(naive_ols.pvalues["log_price"]), 6),
            "ground_truth": -1.5000,
            "abs_error_vs_truth": round(abs(naive_theta - (-1.5)), 4),
        },
        {
            "model": "Member 2 DoWhy (Backdoor)",
            "theta_elasticity": -1.4091,
            "std_error": 0.0850,
            "ci_95_low": -1.5757,
            "ci_95_high": -1.2425,
            "p_value": 0.000000,
            "ground_truth": -1.5000,
            "abs_error_vs_truth": round(abs(-1.4091 - (-1.5)), 4),
        },
    ]

    primary_res = None
    for learner_name, label in [
        ("random_forest", "Double ML (Random Forest)"),
        ("gradient_boosting", "Double ML (Gradient Boosting)"),
        ("ridge", "Double ML (Ridge Linear)"),
    ]:
        res = estimate_dml_elasticity(df, learner=learner_name, include_product_fe=True)
        if learner_name == "random_forest":
            primary_res = res
        model_rows.append({
            "model": label,
            "theta_elasticity": round(res["theta_elasticity"], 4),
            "std_error": round(res["std_error"], 4),
            "ci_95_low": round(res["ci_95_low"], 4),
            "ci_95_high": round(res["ci_95_high"], 4),
            "p_value": round(res["p_value"], 6),
            "ground_truth": -1.5000,
            "abs_error_vs_truth": round(abs(res["theta_elasticity"] - (-1.5)), 4),
        })

    comparison_df = pd.DataFrame(model_rows)
    comparison_df.to_csv(os.path.join(out_dir, "dml_model_comparison.csv"), index=False)

    # Save orthogonalized residuals for diagnostics & dashboard
    residuals_df = df[
        [
            "store_id",
            "product_id",
            "product_name",
            "week",
            "avg_unit_price",
            "total_units_sold",
            "total_revenue",
            "is_treated_store",
            "is_post_treatment",
            "treatment_active",
            "log_price",
            "log_units",
        ]
    ].copy()
    residuals_df["log_units_residual"] = primary_res["y_res"]
    residuals_df["log_price_residual"] = primary_res["t_res"]
    residuals_df.to_csv(os.path.join(out_dir, "dml_orthogonal_residuals.csv"), index=False)

    # CATE table
    cate_df = estimate_cate_by_segment(df)
    cate_df.to_csv(os.path.join(out_dir, "dml_cate_segments.csv"), index=False)

    # Counterfactual impact on Store_1 PROD_101
    cf_df = compute_intervention_counterfactual(df, primary_res["theta_elasticity"])
    cf_df.to_csv(os.path.join(out_dir, "store1_prod101_dml_counterfactual.csv"), index=False)

    post_cf = cf_df[cf_df["week"] >= 36]
    summary = {
        "primary_model": "Double ML (Random Forest, 5-Fold Cross-Fitting)",
        "ground_truth_elasticity": -1.5000,
        "member2_dowhy_elasticity": -1.4091,
        "dml_theta_elasticity": round(primary_res["theta_elasticity"], 4),
        "dml_std_error": round(primary_res["std_error"], 4),
        "dml_ci_95": [round(primary_res["ci_95_low"], 4), round(primary_res["ci_95_high"], 4)],
        "naive_ols_elasticity": round(naive_theta, 4),
        "first_stage_r2_outcome": round(primary_res["first_stage_r2_y"], 4),
        "first_stage_r2_treatment": round(primary_res["first_stage_r2_t"], 4),
        "post_intervention_avg_price_change_pct": round(float(post_cf["price_delta_pct"].mean()), 2),
        "post_intervention_causal_units_lift_pct": round(float(post_cf["causal_units_lift_pct"].mean()), 2),
        "post_intervention_total_units_delta": round(float(post_cf["causal_units_lift"].sum()), 1),
        "post_intervention_total_revenue_delta": round(float(post_cf["causal_revenue_lift"].sum()), 2),
    }
    with open(os.path.join(out_dir, "dml_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
