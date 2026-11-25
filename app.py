import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from src.dml.double_ml_engine import estimate_dml_elasticity, NUMERIC_CONFOUNDERS

st.set_page_config(
    page_title="Causal Pricing Intelligence Platform | Zaalima Enterprise Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive card polish, clean spacing, and modern SaaS appearance
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
    }
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #1d4ed8 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 18px;
        color: #f8fafc;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 4px;
        color: #ffffff;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-bottom: 0;
    }
    .badge-pill {
        display: inline-block;
        background-color: rgba(56, 189, 248, 0.18);
        color: #7dd3fc;
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 999px;
        padding: 3px 11px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PANEL_PATH = "data/processed/store_product_week_panel.csv"
RESULTS_DIR = "reports/dml_results"
EDA_FIG_DIR = "reports/eda_figures"
DML_FIG_DIR = "reports/dml_figures"


@st.cache_data
def load_artifacts():
    panel = pd.read_csv(PANEL_PATH)
    comp = pd.read_csv(os.path.join(RESULTS_DIR, "dml_model_comparison.csv"))
    cate = pd.read_csv(os.path.join(RESULTS_DIR, "dml_cate_segments.csv"))
    cf = pd.read_csv(os.path.join(RESULTS_DIR, "store1_prod101_dml_counterfactual.csv"))
    res = pd.read_csv(os.path.join(RESULTS_DIR, "dml_orthogonal_residuals.csv"))
    with open(os.path.join(RESULTS_DIR, "dml_summary.json"), "r") as f:
        summary = json.load(f)
    return panel, comp, cate, cf, res, summary


if not os.path.exists(os.path.join(RESULTS_DIR, "dml_summary.json")):
    st.error("DML artifacts not found. Run `python run_member4_dml_pipeline.py` first.")
    st.stop()

panel, comp, cate, cf, res, summary = load_artifacts()

# ==============================================================================
# SIDEBAR: Global Controls, Live Model Workbench & Export Center
# ==============================================================================
with st.sidebar:
    st.header("Control Center")
    st.caption("Global filters & live causal workbench")

    all_products = sorted(panel["product_id"].unique())
    prod_labels = {
        row["product_id"]: f"{row['product_id']} — {row['product_name']}"
        for _, row in panel[["product_id", "product_name"]].drop_duplicates().iterrows()
    }
    selected_product = st.selectbox(
        "Target Product SKU",
        options=all_products,
        format_func=lambda x: prod_labels[x],
        index=0,
    )

    all_stores = sorted(panel["store_id"].unique(), key=lambda s: int(s.split("_")[1]))
    selected_stores = st.multiselect(
        "Active Store Cohort",
        options=all_stores,
        default=all_stores,
    )

    week_range = st.slider(
        "Evaluation Week Window",
        min_value=int(panel["week"].min()),
        max_value=int(panel["week"].max()),
        value=(1, 52),
    )

    st.divider()
    st.subheader("Live DML Model Config")
    live_learner = st.selectbox(
        "First-Stage Nuisance Learner",
        options=["random_forest", "gradient_boosting", "ridge"],
        format_func=lambda x: {
            "random_forest": "Random Forest (200 trees)",
            "gradient_boosting": "Gradient Boosting (150 trees)",
            "ridge": "Ridge Linear CV",
        }[x],
    )
    active_confounders = st.multiselect(
        "Backdoor Confounder Set (X)",
        options=NUMERIC_CONFOUNDERS,
        default=NUMERIC_CONFOUNDERS,
    )

    st.divider()
    st.subheader("Export Results")
    st.download_button(
        label="Download DML Benchmark CSV",
        data=comp.to_csv(index=False).encode("utf-8"),
        file_name="dml_model_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        label="Download CATE Segments CSV",
        data=cate.to_csv(index=False).encode("utf-8"),
        file_name="dml_cate_segments.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        label="Download Counterfactual CSV",
        data=cf.to_csv(index=False).encode("utf-8"),
        file_name="store1_prod101_counterfactual.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Filter panel dynamically based on sidebar selection
filtered_panel = panel[
    (panel["product_id"] == selected_product)
    & (panel["store_id"].isin(selected_stores if selected_stores else all_stores))
    & (panel["week"].between(week_range[0], week_range[1]))
].copy()

# Compute live filtered DML elasticity if user modifies sidebar controls
if len(filtered_panel) >= 30 and len(active_confounders) >= 1:
    live_dml = estimate_dml_elasticity(
        filtered_panel,
        learner=live_learner,
        confounders=active_confounders,
        include_product_fe=False,
    )
else:
    live_dml = {
        "theta_elasticity": summary["dml_theta_elasticity"],
        "std_error": summary["dml_std_error"],
        "ci_95_low": summary["dml_ci_95"][0],
        "ci_95_high": summary["dml_ci_95"][1],
        "p_value": 0.0,
        "first_stage_r2_y": summary["first_stage_r2_outcome"],
        "first_stage_r2_t": summary["first_stage_r2_treatment"],
        "n_obs": len(filtered_panel),
    }

# ==============================================================================
# HERO BANNER & EXECUTIVE KPI STRIP
# ==============================================================================
st.markdown(
    """
    <div class="hero-banner">
        <div>
            <span class="badge-pill">ZAALIMA ENTERPRISE ANALYTICS</span>
            <span class="badge-pill">MEMBER 4: DOUBLE ML &amp; DECISION SUITE</span>
            <span class="badge-pill">CONTRIBUTOR: RIYAZ M</span>
        </div>
        <div class="hero-title">Causal Pricing Intelligence &amp; Elasticity Decision Platform</div>
        <p class="hero-sub">
            Orthogonalized Double Machine Learning (Chernozhukov et al.), Segment CATE Heterogeneity,
            Revenue Optimization Frontier &amp; Full-Team Causal Pipeline Integration (Members 1–4)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(
    "Pooled DML Elasticity (θ)",
    f"{summary['dml_theta_elasticity']:.4f}",
    delta=f"Target: {summary['ground_truth_elasticity']:.2f}",
)
k2.metric(
    f"Live Slice θ ({selected_product})",
    f"{live_dml['theta_elasticity']:.4f}",
    delta=f"SE: ±{live_dml['std_error']:.4f}",
)
k3.metric(
    "Naive OLS (Confounded)",
    f"{summary['naive_ols_elasticity']:.4f}",
    delta="99.2% Bias Error",
    delta_color="inverse",
)
k4.metric(
    "Member 2 DoWhy Check",
    f"{summary['member2_dowhy_elasticity']:.4f}",
    delta="Backdoor Validated",
)
k5.metric(
    "Store_1 Demand Impact",
    f"{summary['post_intervention_causal_units_lift_pct']:.2f}%",
    delta=f"{summary['post_intervention_total_units_delta']:.0f} units (Wks 36–52)",
)
k6.metric(
    "Store_1 Revenue Delta",
    f"${summary['post_intervention_total_revenue_delta']:,.2f}",
    delta="+15.0% Price Hike",
    delta_color="inverse",
)

st.write("")

# ==============================================================================
# 6-TAB COMPREHENSIVE ANALYTICAL WORKSPACE
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Executive Overview & Market Trends",
    "2. Double ML Benchmark & Live Estimator",
    "3. CATE Segment Heterogeneity",
    "4. Pricing Simulator & Profit Optimizer",
    "5. Causal Diagnostics & Residuals",
    "6. Team Architecture & Pipeline Audit",
])

# ------------------------------------------------------------------------------
# TAB 1: Executive Overview & Market Trends
# ------------------------------------------------------------------------------
with tab1:
    st.subheader(f"Market Trajectory & Intervention Monitor — {prod_labels[selected_product]}")
    st.caption(
        "Compare Store_1 (treated store starting Week 36) against the 9-store donor pool across price, unit demand, and gross revenue."
    )

    trend_df = filtered_panel.copy()
    trend_df["cohort"] = np.where(trend_df["store_id"] == "Store_1", "Store_1 (Treated)", "Donor Pool Mean (Store 2–10)")
    agg_trend = (
        trend_df.groupby(["week", "cohort"], as_index=False)
        .agg(
            avg_price=("avg_unit_price", "mean"),
            avg_units=("total_units_sold", "mean"),
            avg_revenue=("total_revenue", "mean"),
            avg_footfall=("avg_store_footfall", "mean"),
        )
    )

    c_left, c_right = st.columns(2)
    with c_left:
        fig_price = px.line(
            agg_trend,
            x="week",
            y="avg_price",
            color="cohort",
            markers=True,
            color_discrete_map={"Store_1 (Treated)": "#ef4444", "Donor Pool Mean (Store 2–10)": "#38bdf8"},
            labels={"week": "Calendar Week", "avg_price": "Average Shelf Price ($)", "cohort": "Store Group"},
            title="Weekly Shelf Price Trajectory ($)",
        )
        fig_price.add_vline(x=35.5, line_dash="dash", line_color="#f59e0b", annotation_text="Week 36 Intervention")
        fig_price.update_layout(height=360, legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_price, use_container_width=True)

    with c_right:
        fig_units = px.line(
            agg_trend,
            x="week",
            y="avg_units",
            color="cohort",
            markers=True,
            color_discrete_map={"Store_1 (Treated)": "#ef4444", "Donor Pool Mean (Store 2–10)": "#38bdf8"},
            labels={"week": "Calendar Week", "avg_units": "Mean Weekly Units Sold", "cohort": "Store Group"},
            title="Weekly Sales Volume Trajectory (Units)",
        )
        fig_units.add_vline(x=35.5, line_dash="dash", line_color="#f59e0b", annotation_text="Week 36 Intervention")
        fig_units.update_layout(height=360, legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_units, use_container_width=True)

    # Store-level performance summary table
    st.markdown("**Store-Level Performance Summary (Filtered Slice)**")
    store_summary = (
        filtered_panel.groupby("store_id", as_index=False)
        .agg(
            weeks_observed=("week", "count"),
            mean_price=("avg_unit_price", "mean"),
            total_units=("total_units_sold", "sum"),
            total_revenue=("total_revenue", "sum"),
            mean_competitor_price=("avg_competitor_price", "mean"),
            mean_footfall=("avg_store_footfall", "mean"),
            total_ad_spend=("total_marketing_spend", "sum"),
        )
        .round(2)
    )
    st.dataframe(store_summary, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 2: Double ML Benchmark & Live Estimator
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Cross-Fitted Double Machine Learning Benchmark vs. Naive OLS & Member 2 DoWhy")
    col_b1, col_b2 = st.columns([1.25, 1])

    with col_b1:
        comp_live = comp.copy()
        live_row = pd.DataFrame([{
            "model": f"Live Slice DML ({selected_product}, {live_learner})",
            "theta_elasticity": round(live_dml["theta_elasticity"], 4),
            "std_error": round(live_dml["std_error"], 4),
            "ci_95_low": round(live_dml["ci_95_low"], 4),
            "ci_95_high": round(live_dml["ci_95_high"], 4),
            "p_value": round(live_dml["p_value"], 6),
            "ground_truth": -1.5000,
            "abs_error_vs_truth": round(abs(live_dml["theta_elasticity"] - (-1.5)), 4),
        }])
        comp_display = pd.concat([comp_live, live_row], ignore_index=True)

        fig_bench = go.Figure()
        for _, row in comp_display.iterrows():
            if "Naive" in row["model"]:
                color = "#94a3b8"
            elif "Member 2" in row["model"]:
                color = "#38bdf8"
            elif "Live Slice" in row["model"]:
                color = "#f59e0b"
            else:
                color = "#f43f5e"

            fig_bench.add_trace(
                go.Scatter(
                    x=[row["theta_elasticity"]],
                    y=[row["model"]],
                    mode="markers+text",
                    text=[f"  {row['theta_elasticity']:.3f}"],
                    textposition="top center",
                    marker=dict(size=12, color=color),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[row["ci_95_high"] - row["theta_elasticity"]],
                        arrayminus=[row["theta_elasticity"] - row["ci_95_low"]],
                        thickness=2.5,
                        width=6,
                        color=color,
                    ),
                    name=row["model"],
                )
            )
        fig_bench.add_vline(x=-1.50, line_dash="dash", line_color="#22c55e", annotation_text="Ground Truth (-1.50)")
        fig_bench.add_vline(x=0.0, line_dash="dot", line_color="#64748b")
        fig_bench.update_layout(
            xaxis_title="Estimated Own-Price Elasticity (θ) with 95% Confidence Interval",
            yaxis=dict(autorange="reversed"),
            showlegend=False,
            height=430,
            margin=dict(l=10, r=20, t=30, b=30),
        )
        st.plotly_chart(fig_bench, use_container_width=True)

    with col_b2:
        st.markdown("**Estimator Comparison & Error Audit**")
        st.dataframe(comp_display, use_container_width=True, hide_index=True)
        st.success(
            f"**Live First-Stage Nuisance Fit ({selected_product}):**  \n"
            f"• Outcome Model $R^2$ (`ln(Q) | X`): **{live_dml['first_stage_r2_y']:.4f}**  \n"
            f"• Treatment Model $R^2$ (`ln(P) | X`): **{live_dml['first_stage_r2_t']:.4f}**  \n"
            f"• Observations Evaluated: **{live_dml['n_obs']}** store-weeks"
        )
        st.info(
            "**Why Naive OLS Fails:** Unadjusted OLS (`-0.0124`) mixes high-price/high-volume flagship drinks "
            "with lower-price/lower-volume items and ignores footfall surges. 5-fold cross-fitted DML removes "
            "confounding nuisance variation first, isolating the true causal slope."
        )

# ------------------------------------------------------------------------------
# TAB 3: CATE Segment Heterogeneity
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Conditional Average Treatment Effects (CATE) Across Products & Stores")
    c_seg1, c_seg2 = st.columns([1, 2.2])
    with c_seg1:
        seg_choice = st.radio(
            "Segment Dimension",
            options=["Product", "Store Group", "Store"],
            index=0,
        )
        st.markdown(
            """
            **Business Interpretation:**
            - **Elastic ($|\theta| > 1$):** Price increases shrink revenue because demand drops proportionally faster than price rises (`PROD_101`, `PROD_102`).
            - **Near-Unit / Inelastic ($|\theta| \le 1$):** Moderate price hikes can preserve or lift gross revenue (`PROD_103`).
            """
        )

    with c_seg2:
        sub_cate = cate[cate["segment_type"] == seg_choice].copy()
        fig_cate = px.bar(
            sub_cate,
            x="theta_elasticity",
            y="segment_value",
            orientation="h",
            error_x=1.96 * sub_cate["std_error"],
            color="theta_elasticity",
            color_continuous_scale="RdBu_r",
            labels={"theta_elasticity": "Estimated Price Elasticity (θ)", "segment_value": seg_choice},
            height=400,
        )
        fig_cate.add_vline(x=-1.50, line_dash="dash", line_color="#22c55e", annotation_text="Ground Truth (-1.50)")
        fig_cate.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cate, use_container_width=True)

    st.dataframe(sub_cate, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 4: Pricing Simulator & Profit Optimizer
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Store_1 Counterfactual Analysis & Interactive Revenue/Margin Optimizer")
    sim_left, sim_right = st.columns([1.05, 2.15])

    with sim_left:
        sim_price_change = st.slider("Proposed Price Adjustment (%)", min_value=-25, max_value=35, value=15, step=1)
        unit_cogs = st.number_input("Unit Cost of Goods Sold — COGS ($)", min_value=0.20, max_value=3.00, value=1.40, step=0.05)
        theta_used = st.selectbox(
            "Causal Elasticity Model",
            options=[
                ("DML Random Forest (-1.2595)", summary["dml_theta_elasticity"]),
                ("DML Ridge Linear (-1.3268)", -1.3268),
                ("PROD_101 Specific CATE (-1.3423)", -1.3423),
                ("Member 2 DoWhy (-1.4091)", summary["member2_dowhy_elasticity"]),
                ("Ground Truth (-1.5000)", -1.5000),
            ],
            format_func=lambda x: x[0],
        )[1]

        base_p = 3.50
        base_q = float(cf[cf["week"] <= 35]["total_units_sold"].mean())
        new_p = base_p * (1.0 + sim_price_change / 100.0)
        new_q = base_q * ((new_p / base_p) ** theta_used)

        base_rev = base_p * base_q
        new_rev = new_p * new_q
        base_profit = (base_p - unit_cogs) * base_q
        new_profit = (new_p - unit_cogs) * new_q

        m1, m2 = st.columns(2)
        m1.metric("New Shelf Price", f"${new_p:.2f}", delta=f"{sim_price_change:+d}%")
        m2.metric("Weekly Units", f"{new_q:.0f}", delta=f"{((new_q / base_q) - 1) * 100:+.2f}%")

        m3, m4 = st.columns(2)
        m3.metric("Weekly Revenue", f"${new_rev:,.2f}", delta=f"{((new_rev / base_rev) - 1) * 100:+.2f}%")
        m4.metric("Weekly Gross Profit", f"${new_profit:,.2f}", delta=f"{((new_profit / base_profit) - 1) * 100:+.2f}%")

    with sim_right:
        # Build Revenue & Profit Optimization Curve across -25% to +35% price changes
        pct_grid = np.linspace(-25, 35, 61)
        p_grid = base_p * (1.0 + pct_grid / 100.0)
        q_grid = base_q * ((p_grid / base_p) ** theta_used)
        rev_grid = p_grid * q_grid
        prof_grid = (p_grid - unit_cogs) * q_grid

        opt_idx = int(np.argmax(prof_grid))
        opt_pct = float(pct_grid[opt_idx])
        opt_price = float(p_grid[opt_idx])

        fig_opt = make_subplots(specs=[[{"secondary_y": True}]])
        fig_opt.add_trace(
            go.Scatter(x=pct_grid, y=rev_grid, name="Projected Weekly Revenue ($)", line=dict(color="#38bdf8", width=2.8)),
            secondary_y=False,
        )
        fig_opt.add_trace(
            go.Scatter(x=pct_grid, y=prof_grid, name="Projected Weekly Gross Profit ($)", line=dict(color="#22c55e", width=2.8)),
            secondary_y=True,
        )
        fig_opt.add_vline(x=sim_price_change, line_dash="dash", line_color="#f43f5e", annotation_text=f"Selected ({sim_price_change:+d}%)")
        fig_opt.add_vline(x=opt_pct, line_dash="dot", line_color="#22c55e", annotation_text=f"Profit Peak ({opt_pct:+.0f}% / ${opt_price:.2f})")
        fig_opt.update_layout(
            title=f"Pricing Optimization Frontier (θ = {theta_used:.4f}, Unit COGS = ${unit_cogs:.2f})",
            xaxis_title="Price Change vs. $3.50 Baseline (%)",
            height=380,
            legend=dict(orientation="h", y=1.12),
        )
        fig_opt.update_yaxes(title_text="Weekly Revenue ($)", secondary_y=False)
        fig_opt.update_yaxes(title_text="Weekly Gross Profit ($)", secondary_y=True)
        st.plotly_chart(fig_opt, use_container_width=True)

    st.divider()
    st.markdown("**Historical Store_1 (PROD_101): Actual vs. DML Counterfactual Trajectory (Weeks 1–52)**")
    fig_cf = go.Figure()
    fig_cf.add_trace(go.Scatter(x=cf["week"], y=cf["total_units_sold"], mode="lines+markers", name="Actual Units Sold", line=dict(color="#f43f5e", width=2.5)))
    fig_cf.add_trace(go.Scatter(x=cf["week"], y=cf["counterfactual_units"], mode="lines", name="DML Counterfactual (No +15% Hike)", line=dict(color="#38bdf8", width=2.5, dash="dash")))
    fig_cf.add_vline(x=35.5, line_dash="dot", line_color="#f59e0b", annotation_text="Week 36 Price Hike")
    fig_cf.update_layout(xaxis_title="Calendar Week (1–52)", yaxis_title="Weekly Units Sold", height=350)
    st.plotly_chart(fig_cf, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: Causal Diagnostics & Residuals
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("Frisch-Waugh-Lovell Orthogonalized Residuals & Confounder Correlation Diagnostics")
    d1, d2 = st.columns([1.3, 1])

    with d1:
        sample_res = res.sample(min(750, len(res)), random_state=42)
        fig_res = px.scatter(
            sample_res,
            x="log_price_residual",
            y="log_units_residual",
            color=sample_res["treatment_active"].map({0: "Unshocked Store-Weeks", 1: "Treated Store_1 PROD_101 (Wks 36–52)"}),
            color_discrete_map={"Unshocked Store-Weeks": "#64748b", "Treated Store_1 PROD_101 (Wks 36–52)": "#f43f5e"},
            hover_data=["store_id", "product_id", "week", "avg_unit_price", "total_units_sold"],
            labels={
                "log_price_residual": "Orthogonalized Log Price Residual (T̃)",
                "log_units_residual": "Orthogonalized Log Units Residual (Ỹ)",
                "color": "Cohort",
            },
            title="Second-Stage DML Orthogonalized Fit (Confounders Partialled Out)",
            height=410,
        )
        x_line = np.linspace(sample_res["log_price_residual"].min(), sample_res["log_price_residual"].max(), 50)
        fig_res.add_trace(
            go.Scatter(
                x=x_line,
                y=summary["dml_theta_elasticity"] * x_line,
                mode="lines",
                name=f"DML Slope (θ = {summary['dml_theta_elasticity']:.4f})",
                line=dict(color="#22c55e", width=2.5),
            )
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with d2:
        corr_cols = [
            "avg_unit_price",
            "total_units_sold",
            "avg_competitor_price",
            "total_marketing_spend",
            "avg_store_footfall",
            "holiday_days",
        ]
        corr_mat = filtered_panel[corr_cols].corr().round(2)
        fig_corr = px.imshow(
            corr_mat,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=f"Confounder Correlation Matrix ({selected_product})",
            height=410,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("**Balanced Panel Data Explorer**")
    st.dataframe(filtered_panel, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 6: Team Architecture & Pipeline Audit
# ------------------------------------------------------------------------------
with tab6:
    st.subheader("End-to-End 4-Member Causal Pricing Architecture (Zaalima Development)")
    team_df = pd.DataFrame([
        {
            "Role": "Member 1 — Data Engineering",
            "Contributor": "Samiksha Dewase",
            "Core Methodology": "Transaction Cleaning, Imputation & Balanced Store-Product-Week Panel",
            "Key Artifact": "data/processed/store_product_week_panel.csv (1,560 rows)",
            "Status": "Integrated",
        },
        {
            "Role": "Member 2 — Causal Design & DAG",
            "Contributor": "Srilekha",
            "Core Methodology": "Structural Causal DAG, Backdoor Identification & DoWhy Refuters",
            "Key Artifact": "Minimal Adjustment Set (4 confounders) | θ = -1.4091",
            "Status": "Benchmarked",
        },
        {
            "Role": "Member 3 — Synthetic Control",
            "Contributor": "Omkar Jasud",
            "Core Methodology": "Convex Donor Pool Weights (Store_2–10) & Placebo Permutation Tests",
            "Key Artifact": "Pre-treatment Weeks 1–35 vs. Post-treatment Weeks 36–52",
            "Status": "Aligned",
        },
        {
            "Role": "Member 4 — Double ML & Dashboard",
            "Contributor": "Riyaz M",
            "Core Methodology": "5-Fold Cross-Fitted Partialling-Out DML, CATE & Profit Optimizer UI",
            "Key Artifact": "src/dml/double_ml_engine.py & app.py (θ = -1.2595)",
            "Status": "Active",
        },
    ])
    st.dataframe(team_df, use_container_width=True, hide_index=True)

    st.markdown("**Publication-Ready Static Report Figures (Members 1 & 4)**")
    fig_col1, fig_col2 = st.columns(2)
    p_bench = os.path.join(DML_FIG_DIR, "07_dml_elasticity_benchmark.png")
    p_res = os.path.join(DML_FIG_DIR, "08_dml_orthogonal_residuals.png")
    p_cate = os.path.join(DML_FIG_DIR, "09_dml_cate_heterogeneity.png")
    p_cf = os.path.join(DML_FIG_DIR, "10_dml_counterfactual_impact.png")

    if os.path.exists(p_bench):
        fig_col1.image(p_bench, caption="Figure 07: Double ML Elasticity Benchmark", use_container_width=True)
    if os.path.exists(p_res):
        fig_col2.image(p_res, caption="Figure 08: Orthogonalized Second-Stage Residuals", use_container_width=True)
    if os.path.exists(p_cate):
        fig_col1.image(p_cate, caption="Figure 09: Segment CATE Heterogeneity", use_container_width=True)
    if os.path.exists(p_cf):
        fig_col2.image(p_cf, caption="Figure 10: Store_1 Counterfactual Demand & Revenue Impact", use_container_width=True)