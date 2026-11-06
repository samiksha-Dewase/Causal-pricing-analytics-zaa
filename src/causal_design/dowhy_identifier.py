"""
dowhy_identifier.py
===================
Executes Microsoft DoWhy causal effect identification and sensitivity
refutation tests on the retail panel dataset.

Author: Member 2 (Causal Design & DAG Lead)
Zaalima Development - Enterprise Data Analytics
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dowhy import CausalModel

try:
    from .dag_builder import get_dag_dot_string
except ImportError:
    from dag_builder import get_dag_dot_string


def run_causal_identification_pipeline(
    panel_csv_path: str = "data/processed/store_product_week_panel.csv",
    product_filter: str = "PROD_101",
    treatment_var: str = "log_price",
    outcome_var: str = "log_units",
    figures_dir: str = "reports/causal_figures"
) -> dict:
    """
    Loads the panel dataset, compiles the DoWhy CausalModel, identifies the
    causal estimand via the Backdoor Criterion, computes baseline linear estimate,
    and runs three rigorous causal refutation tests.
    """
    if not os.path.exists(panel_csv_path):
        raise FileNotFoundError(f"Panel dataset not found at: {panel_csv_path}")

    df = pd.read_csv(panel_csv_path)
    if product_filter:
        df = df[df["product_id"] == product_filter].copy()

    dot_graph = get_dag_dot_string()

    print(f"[DoWhy Pipeline] Ingested panel dataset ({len(df)} rows for {product_filter})")
    print(f"                 Treatment: {treatment_var} | Outcome: {outcome_var}")

    # 1. Initialize DoWhy Causal Model
    model = CausalModel(
        data=df,
        treatment=treatment_var,
        outcome=outcome_var,
        graph=dot_graph.replace("\n", " ")
    )

    # 2. Identify Causal Estimand
    print("[DoWhy Pipeline] Evaluating Backdoor Criterion identification...")
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    backdoor_vars = identified_estimand.get_backdoor_variables()
    print(f"[OK] Backdoor Adjustment Set Identified: {backdoor_vars}")

    # 3. Initial Baseline Causal Estimation
    print("[DoWhy Pipeline] Estimating baseline treatment effect via Backdoor Linear Regression...")
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=False
    )
    estimated_elasticity = estimate.value
    print(f"[OK] Initial Estimated Price Elasticity (eta): {round(estimated_elasticity, 4)}")
    print(f"     Ground Truth Target Benchmark:            -1.5000")

    # 4. Sensitivity & Refutation Tests
    print("\n[DoWhy Pipeline] Running Causal Robustness & Refutation Tests...")

    # Refutation 1: Add Random Common Cause
    print("  -> Running Test 1: Random Common Cause Refuter...")
    ref_random = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="random_common_cause",
        num_simulations=5,
        random_state=42
    )

    # Refutation 2: Placebo Treatment Refuter
    print("  -> Running Test 2: Placebo Treatment Refuter...")
    ref_placebo = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
        num_simulations=5,
        random_state=42
    )

    # Refutation 3: Data Subset Refuter
    print("  -> Running Test 3: Data Subset Refuter (Subset Fraction: 85%)...")
    ref_subset = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="data_subset_refuter",
        subset_fraction=0.85,
        num_simulations=5,
        random_state=42
    )

    # 5. Render Diagnostics Plot
    os.makedirs(figures_dir, exist_ok=True)
    diag_fig_path = os.path.join(figures_dir, "06_refutation_diagnostics.png")

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    tests = ["Baseline Estimate", "Random Common Cause", "Data Subset (85%)", "Placebo Treatment"]
    values = [
        estimated_elasticity,
        ref_random.new_effect,
        ref_subset.new_effect,
        ref_placebo.new_effect
    ]
    colors = ["#2563eb", "#0d9488", "#059669", "#dc2626"]

    bars = ax.bar(tests, values, color=colors, width=0.48, edgecolor="#1e293b", linewidth=1.2)
    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1.0)
    ax.axhline(-1.50, color="#2563eb", linestyle=":", linewidth=1.8, label="Ground Truth Elasticity (η = -1.50)")

    for bar, val in zip(bars, values):
        y_pos = val - 0.12 if val < 0 else val + 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:.3f}", ha="center", va="center", fontweight="bold", fontsize=10)

    ax.set_title("DoWhy Causal Robustness & Refutation Audit", fontsize=13, fontweight="bold", pad=15, color="#0f172a")
    ax.set_ylabel("Estimated Treatment Elasticity (η)", fontsize=11, fontweight="bold")
    ax.set_ylim(-2.1, 0.4)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(diag_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Refutation diagnostics figure saved to {diag_fig_path}")

    results = {
        "identified_estimand": identified_estimand,
        "backdoor_variables": backdoor_vars,
        "baseline_estimate": estimated_elasticity,
        "ref_random_effect": ref_random.new_effect,
        "ref_placebo_effect": ref_placebo.new_effect,
        "ref_subset_effect": ref_subset.new_effect,
        "diagnostics_figure": diag_fig_path
    }
    return results


if __name__ == "__main__":
    run_causal_identification_pipeline()
