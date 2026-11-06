"""
dag_builder.py
==============
Constructs, verifies, and renders the Directed Acyclic Graph (DAG) for
the Algorithmic Pricing Engine using NetworkX and DOT notation for DoWhy.

Author: Member 2 (Causal Design & DAG Lead)
Zaalima Development - Enterprise Data Analytics
"""

import os
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_dag_dot_string() -> str:
    """
    Returns the DOT representation of the causal DAG for DoWhy ingestion.
    """
    dot_str = """digraph {
        log_competitor_price -> log_price;
        log_competitor_price -> log_units;
        total_marketing_spend -> log_price;
        total_marketing_spend -> log_units;
        avg_store_footfall -> log_price;
        avg_store_footfall -> log_units;
        holiday_days -> log_price;
        holiday_days -> log_units;
        log_price -> log_units;
        log_price -> log_revenue;
        log_units -> log_revenue;
    }"""
    return dot_str


def build_causal_dag() -> nx.DiGraph:
    """
    Builds a NetworkX directed graph encoding structural causal domain knowledge.
    Asserts graph acyclicity.
    """
    dag = nx.DiGraph()

    # Backdoor confounding edges
    dag.add_edge("log_competitor_price", "log_price")
    dag.add_edge("log_competitor_price", "log_units")

    dag.add_edge("total_marketing_spend", "log_price")
    dag.add_edge("total_marketing_spend", "log_units")

    dag.add_edge("avg_store_footfall", "log_price")
    dag.add_edge("avg_store_footfall", "log_units")

    dag.add_edge("holiday_days", "log_price")
    dag.add_edge("holiday_days", "log_units")

    # Primary causal channel (Own-Price Elasticity eta = -1.50)
    dag.add_edge("log_price", "log_units")

    # Downstream accounting mediator
    dag.add_edge("log_price", "log_revenue")
    dag.add_edge("log_units", "log_revenue")

    assert nx.is_directed_acyclic_graph(dag), "Validation Error: Causal graph contains cycles!"
    return dag


def plot_and_export_dag(output_path: str = "reports/causal_figures/05_causal_dag_structure.png") -> str:
    """
    Renders an executive publication-grade visualization of the Causal DAG.
    """
    dag = build_causal_dag()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Node styling
    node_colors = []
    node_sizes = []
    for node in dag.nodes():
        if node == "log_price":
            node_colors.append("#2563eb")  # Treatment: Deep Blue
            node_sizes.append(4200)
        elif node == "log_units":
            node_colors.append("#16a34a")  # Outcome: Emerald Green
            node_sizes.append(4200)
        elif node == "log_revenue":
            node_colors.append("#9333ea")  # Mediator: Purple
            node_sizes.append(3500)
        else:
            node_colors.append("#ea580c")  # Confounder: Amber
            node_sizes.append(3800)

    # Clean manual 2D coordinates for visual clarity
    pos = {
        "log_competitor_price": (-1.8, 1.8),
        "total_marketing_spend": (-0.6, 2.0),
        "avg_store_footfall": (0.6, 2.0),
        "holiday_days": (1.8, 1.8),
        "log_price": (-1.0, 0.0),
        "log_units": (1.0, 0.0),
        "log_revenue": (0.0, -1.5),
    }

    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Draw nodes
    nx.draw_networkx_nodes(
        dag, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.95,
        edgecolors="#1e293b",
        linewidths=2.2
    )

    # Labels
    labels = {
        "log_competitor_price": "Competitor Price\n(ln P_comp)",
        "total_marketing_spend": "Marketing Spend\n(Mktg)",
        "avg_store_footfall": "Store Footfall\n(Footfall)",
        "holiday_days": "Holiday Days\n(Holidays)",
        "log_price": "Shelf Price\n[Treatment T]\n(ln Price)",
        "log_units": "Sales Units\n[Outcome Y]\n(ln Units)",
        "log_revenue": "Gross Revenue\n[Mediator]\n(ln Revenue)",
    }
    nx.draw_networkx_labels(dag, pos, labels=labels, ax=ax, font_size=8.5, font_weight="bold", font_color="#ffffff")

    # Directed curved edges
    nx.draw_networkx_edges(
        dag, pos, ax=ax,
        edge_color="#475569",
        arrows=True,
        arrowsize=22,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.07",
        width=2.0,
        node_size=node_sizes
    )

    # Emphasize primary causal edge (Price -> Units)
    nx.draw_networkx_edges(
        dag, pos, ax=ax,
        edgelist=[("log_price", "log_units")],
        edge_color="#2563eb",
        arrows=True,
        arrowsize=26,
        arrowstyle="-|>",
        width=3.5,
        node_size=node_sizes
    )
    ax.text(0.0, 0.12, "Causal Elasticity (η = -1.50)", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1d4ed8")

    # Title & Legend
    ax.set_title("Causal Directed Acyclic Graph (DAG): Algorithmic Pricing Engine", fontsize=14, fontweight="bold", pad=20, color="#0f172a")

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Treatment (T: Shelf Price)', markerfacecolor='#2563eb', markersize=13),
        plt.Line2D([0], [0], marker='o', color='w', label='Outcome (Y: Units Sold)', markerfacecolor='#16a34a', markersize=13),
        plt.Line2D([0], [0], marker='o', color='w', label='Backdoor Confounders (Must Adjust)', markerfacecolor='#ea580c', markersize=13),
        plt.Line2D([0], [0], marker='o', color='w', label='Downstream Mediator (Bad Control)', markerfacecolor='#9333ea', markersize=13),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9.5, facecolor="#f8fafc", edgecolor="#cbd5e1")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


if __name__ == "__main__":
    print("[DAG Builder] Compiling Causal DAG...")
    dag = build_causal_dag()
    print(f"[OK] DAG validated: {len(dag.nodes)} nodes, {len(dag.edges)} edges.")
    path = plot_and_export_dag()
    print(f"[OK] Figure saved to {path}")
