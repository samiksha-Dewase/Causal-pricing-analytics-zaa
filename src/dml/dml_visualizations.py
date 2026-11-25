import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def generate_dml_figures(results_dir: str = "reports/dml_results", fig_dir: str = "reports/dml_figures") -> list:
    os.makedirs(fig_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    comp_df = pd.read_csv(os.path.join(results_dir, "dml_model_comparison.csv"))
    res_df = pd.read_csv(os.path.join(results_dir, "dml_orthogonal_residuals.csv"))
    cate_df = pd.read_csv(os.path.join(results_dir, "dml_cate_segments.csv"))
    cf_df = pd.read_csv(os.path.join(results_dir, "store1_prod101_dml_counterfactual.csv"))

    saved_paths = []

    # Figure 07: Model Comparison Forest Plot (Naive OLS vs Member 2 DoWhy vs Member 4 DML Learners)
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)
    y_pos = np.arange(len(comp_df))
    colors = ["#95a5a6" if "Naive" in m else ("#2980b9" if "Member 2" in m else "#c0392b") for m in comp_df["model"]]

    for idx, row in comp_df.iterrows():
        ax.errorbar(
            row["theta_elasticity"],
            idx,
            xerr=[[row["theta_elasticity"] - row["ci_95_low"]], [row["ci_95_high"] - row["theta_elasticity"]]],
            fmt="o",
            color=colors[idx],
            ecolor=colors[idx],
            elinewidth=2.2,
            capsize=4,
            markersize=8,
        )
        ax.text(
            row["ci_95_high"] + 0.03,
            idx,
            f"θ = {row['theta_elasticity']:.3f} [{row['ci_95_low']:.2f}, {row['ci_95_high']:.2f}]",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#2c3e50",
        )

    ax.axvline(-1.50, color="#27ae60", linestyle="--", linewidth=2, label="True Own-Price Elasticity (η = -1.50)")
    ax.axvline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comp_df["model"], fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(-1.95, 0.80)
    ax.set_xlabel("Estimated Own-Price Elasticity (θ) with 95% Confidence Interval", fontsize=10.5)
    ax.set_title("Member 4: Double Machine Learning Elasticity Benchmark vs. Naive OLS & DoWhy", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="center right", frameon=True)
    plt.tight_layout()
    p1 = os.path.join(fig_dir, "07_dml_elasticity_benchmark.png")
    plt.savefig(p1)
    plt.close(fig)
    saved_paths.append(p1)

    # Figure 08: Orthogonalized Residuals Scatter (Frisch-Waugh-Lovell / DML Second Stage)
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=180)
    control_mask = res_df["treatment_active"] == 0
    treated_mask = res_df["treatment_active"] == 1

    ax.scatter(
        res_df.loc[control_mask, "log_price_residual"],
        res_df.loc[control_mask, "log_units_residual"],
        alpha=0.35,
        s=24,
        color="#34495e",
        label="Unshocked Store-Weeks (n = 1,543)",
    )
    ax.scatter(
        res_df.loc[treated_mask, "log_price_residual"],
        res_df.loc[treated_mask, "log_units_residual"],
        alpha=0.90,
        s=55,
        color="#e74c3c",
        edgecolors="black",
        linewidth=0.6,
        label="Treated Store_1 PROD_101 Weeks 36–52 (n = 17)",
    )

    x_grid = np.linspace(res_df["log_price_residual"].min(), res_df["log_price_residual"].max(), 100)
    theta_rf = float(comp_df.loc[comp_df["model"] == "Double ML (Random Forest)", "theta_elasticity"].iloc[0])
    ax.plot(x_grid, theta_rf * x_grid, color="#c0392b", linewidth=2.4, label=f"DML Orthogonal Slope (θ = {theta_rf:.4f})")
    ax.plot(x_grid, -1.50 * x_grid, color="#27ae60", linestyle="--", linewidth=1.8, label="Ground Truth Slope (η = -1.5000)")

    ax.set_xlabel("Orthogonalized Log Price Residual: T̃ = ln(P) − E[ln(P) | X]", fontsize=10.5)
    ax.set_ylabel("Orthogonalized Log Units Residual: Ỹ = ln(Q) − E[ln(Q) | X]", fontsize=10.5)
    ax.set_title("Double ML Second-Stage Orthogonalized Elasticity Fit (Confounders Partialled Out)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    p2 = os.path.join(fig_dir, "08_dml_orthogonal_residuals.png")
    plt.savefig(p2)
    plt.close(fig)
    saved_paths.append(p2)

    # Figure 09: CATE Heterogeneity Across Products & Stores
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)

    prod_cate = cate_df[cate_df["segment_type"].isin(["Product", "Store Group"])].copy()
    axes[0].barh(prod_cate["segment_value"], prod_cate["theta_elasticity"], color=["#2980b9", "#3498db", "#5dade2", "#e74c3c", "#7f8c8d"], alpha=0.9)
    axes[0].errorbar(
        prod_cate["theta_elasticity"],
        np.arange(len(prod_cate)),
        xerr=1.96 * prod_cate["std_error"],
        fmt="none",
        ecolor="black",
        elinewidth=1.4,
        capsize=3,
    )
    axes[0].axvline(-1.50, color="#27ae60", linestyle="--", linewidth=1.6, label="Ground Truth (-1.50)")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Estimated Price Elasticity (θ)")
    axes[0].set_title("CATE by Product SKU & Store Group", fontsize=11, fontweight="bold")
    axes[0].legend(loc="lower left")

    store_cate = cate_df[cate_df["segment_type"] == "Store"].copy()
    store_colors = ["#e74c3c" if s == "Store_1" else "#34495e" for s in store_cate["segment_value"]]
    axes[1].barh(store_cate["segment_value"], store_cate["theta_elasticity"], color=store_colors, alpha=0.88)
    axes[1].errorbar(
        store_cate["theta_elasticity"],
        np.arange(len(store_cate)),
        xerr=1.96 * store_cate["std_error"],
        fmt="none",
        ecolor="black",
        elinewidth=1.3,
        capsize=3,
    )
    axes[1].axvline(-1.50, color="#27ae60", linestyle="--", linewidth=1.6, label="Ground Truth (-1.50)")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Estimated Price Elasticity (θ)")
    axes[1].set_title("Store-Level CATE Heterogeneity (Store_1 Highlighted)", fontsize=11, fontweight="bold")
    axes[1].legend(loc="lower left")

    plt.suptitle("Conditional Average Treatment Effects (CATE) Across Retail Segments", fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    p3 = os.path.join(fig_dir, "09_dml_cate_heterogeneity.png")
    plt.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p3)

    # Figure 10: Store_1 PROD_101 Actual vs DML Counterfactual Demand & Revenue Impact
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), dpi=180, sharex=True)

    axes[0].plot(cf_df["week"], cf_df["total_units_sold"], color="#c0392b", linewidth=2.3, marker="o", markersize=4, label="Actual Units Sold (Store_1, PROD_101)")
    axes[0].plot(cf_df["week"], cf_df["counterfactual_units"], color="#2980b9", linestyle="--", linewidth=2.2, label="DML Counterfactual Units (No +15% Price Hike)")
    axes[0].fill_between(
        cf_df["week"],
        cf_df["total_units_sold"],
        cf_df["counterfactual_units"],
        where=(cf_df["week"] >= 36),
        color="#e74c3c",
        alpha=0.18,
        label="Causal Demand Loss (-16.1% avg)",
    )
    axes[0].axvline(35.5, color="black", linestyle=":", linewidth=1.6, label="Week 36 Price Hike (+15%)")
    axes[0].set_ylabel("Weekly Units Sold", fontsize=10.5)
    axes[0].set_title("Store_1 (PROD_101): Actual vs. DML Counterfactual Demand & Revenue Trajectory", fontsize=12, fontweight="bold", pad=10)
    axes[0].legend(loc="upper right", frameon=True)

    axes[1].plot(cf_df["week"], cf_df["total_revenue"], color="#16a085", linewidth=2.3, marker="s", markersize=4, label="Actual Weekly Revenue ($)")
    axes[1].plot(cf_df["week"], cf_df["counterfactual_revenue"], color="#7f8c8d", linestyle="--", linewidth=2.2, label="DML Counterfactual Revenue ($)")
    axes[1].axvline(35.5, color="black", linestyle=":", linewidth=1.6)
    axes[1].set_xlabel("Week of Year (1 – 52)", fontsize=10.5)
    axes[1].set_ylabel("Weekly Gross Revenue ($)", fontsize=10.5)
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    p4 = os.path.join(fig_dir, "10_dml_counterfactual_impact.png")
    plt.savefig(p4)
    plt.close(fig)
    saved_paths.append(p4)

    return saved_paths
