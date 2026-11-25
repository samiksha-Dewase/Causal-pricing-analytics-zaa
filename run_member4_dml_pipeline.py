import os
from src.dml.double_ml_engine import run_full_dml_analysis
from src.dml.dml_visualizations import generate_dml_figures


def main():
    panel_path = "data/processed/store_product_week_panel.csv"
    results_dir = "reports/dml_results"
    fig_dir = "reports/dml_figures"

    if not os.path.exists(panel_path):
        raise FileNotFoundError(
            f"Panel dataset missing at {panel_path}. Run `python run_member1_pipeline.py` first."
        )

    print("=== Running Member 4: Double Machine Learning (DML) Pipeline ===")
    summary = run_full_dml_analysis(panel_path=panel_path, out_dir=results_dir)
    figs = generate_dml_figures(results_dir=results_dir, fig_dir=fig_dir)

    print(f"[OK] Primary DML Elasticity (Random Forest): {summary['dml_theta_elasticity']:.4f} "
          f"(95% CI: [{summary['dml_ci_95'][0]:.4f}, {summary['dml_ci_95'][1]:.4f}])")
    print(f"[OK] Naive Unadjusted OLS Elasticity:        {summary['naive_ols_elasticity']:.4f}")
    print(f"[OK] Ground-Truth Target Elasticity:         {summary['ground_truth_elasticity']:.4f}")
    print(f"[OK] Post-Intervention Causal Units Lift:    {summary['post_intervention_causal_units_lift_pct']:.2f}% "
          f"({summary['post_intervention_total_units_delta']:.1f} units)")
    print(f"[OK] Post-Intervention Causal Revenue Delta: ${summary['post_intervention_total_revenue_delta']:,.2f}")
    print(f"[OK] Saved {len(figs)} publication figures to {fig_dir}/")
    print("=== Member 4 Pipeline Complete. Launch dashboard via: streamlit run app.py ===")


if __name__ == "__main__":
    main()
