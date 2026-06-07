#!/usr/bin/env python
"""
Generate visualization plots for feature importance analysis.
Supports both ITM-Rec and OULAD datasets.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

output_dir = Path(ROOT).parent / "results" / "feature_importance_analysis"
# Alternative: if scripts folder is the cwd
if not output_dir.exists():
    output_dir = ROOT.parent / "results" / "feature_importance_analysis"
# Final fallback
if not output_dir.exists():
    output_dir = Path.cwd() / "results" / "feature_importance_analysis"
output_dir.mkdir(parents=True, exist_ok=True)


def create_importance_visualizations(dataset_type: str):
    """Create visualization plots for a specific dataset."""
    print(f"\nGenerating visualizations for {dataset_type.upper()}...")
    
    # Load DQN results
    results_file = output_dir / f"dqn_state_importance_{dataset_type}.csv"
    if not results_file.exists():
        print(f"  Warning: {results_file} not found, skipping...")
        return
    
    dqn_results = pd.read_csv(results_file)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Feature Importance Analysis: DQN State Components ({dataset_type.upper()})", 
                 fontsize=18, fontweight="bold", y=0.995)
    
    # 1. DQN State Importance
    ax = axes[0, 0]
    components = dqn_results["component"].values
    importances = dqn_results["importance"].values
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(components)))
    
    sorted_idx = np.argsort(importances)
    ax.barh(range(len(sorted_idx)), importances[sorted_idx], color=colors[sorted_idx], 
            edgecolor="black", linewidth=1.5)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(components[sorted_idx], fontweight="bold")
    ax.set_xlabel("Gradient Magnitude (Normalized)", fontsize=12, fontweight="bold")
    ax.set_title(f"DQN State Importance for Q(s, a*) [{dataset_type.upper()}]", 
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    
    # Add value labels
    for i, (idx, val) in enumerate(zip(sorted_idx, importances[sorted_idx])):
        ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontweight="bold")
    
    # 2. DQN Component Breakdown (detailed)
    ax = axes[0, 1]
    n_components = dqn_results["n_components"].values
    importance_per_component = importances / n_components
    
    sorted_idx2 = np.argsort(importance_per_component)
    colors2 = plt.cm.viridis(np.linspace(0, 1, len(components)))
    ax.barh(range(len(sorted_idx2)), importance_per_component[sorted_idx2], 
            color=colors2[sorted_idx2], edgecolor="black", linewidth=1.5)
    ax.set_yticks(range(len(sorted_idx2)))
    ax.set_yticklabels(components[sorted_idx2], fontweight="bold")
    ax.set_xlabel("Average Importance per Component", fontsize=12, fontweight="bold")
    ax.set_title("DQN Importance Normalized by Component Count", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    
    for i, (idx, val) in enumerate(zip(sorted_idx2, importance_per_component[sorted_idx2])):
        ax.text(val + 0.005, i, f"{val:.4f}", va="center", fontweight="bold")
    
    # 3. Feature distribution summary
    ax = axes[1, 0]
    importance_dict = {row["component"]: row["importance"] for _, row in dqn_results.iterrows()}
    categories = list(importance_dict.keys())
    values = list(importance_dict.values())
    
    wedges, texts, autotexts = ax.pie(values, labels=categories, autopct="%1.1f%%",
                                        colors=plt.cm.Set3(range(len(categories))),
                                        startangle=90, textprops={"fontweight": "bold", "fontsize": 10})
    ax.set_title(f"DQN State Composition by Importance ({dataset_type.upper()})", 
                 fontsize=13, fontweight="bold")
    
    # 4. Summary statistics table
    ax = axes[1, 1]
    ax.axis("tight")
    ax.axis("off")
    
    summary_data = []
    for _, row in dqn_results.iterrows():
        summary_data.append([
            row["component"],
            f"{row['importance']:.4f}",
            int(row["n_components"]),
            row["description"][:45] + "..." if len(row["description"]) > 45 else row["description"]
        ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=["Component", "Importance", "Dims", "Description"],
                    cellLoc="left",
                    loc="center",
                    colWidths=[0.2, 0.15, 0.1, 0.55])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#E7E6E6")
            else:
                table[(i, j)].set_facecolor("#F2F2F2")
    
    ax.set_title("Summary Statistics", fontsize=13, fontweight="bold", pad=20)
    
    plt.tight_layout()
    output_file = output_dir / f"feature_importance_visualization_{dataset_type}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_architecture_comparison_plots():
    """Create architecture comparison plots for both datasets."""
    print("\nGenerating architecture comparison plots...")
    
    # Create a figure with 2x2 subplots (both datasets)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Model Architectures: Input Features Comparison (Both Datasets)", 
                 fontsize=18, fontweight="bold")
    
    # ITM-Rec DeepFM
    ax = axes[0, 0]
    deepfm_categories = ["Embeddings\n(128)", "Context\n(8)", "Linear\n(3)", "FM\n(pairwise)"]
    deepfm_values = [128, 8, 3, 1]
    colors_deepfm = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(deepfm_categories, deepfm_values, color=colors_deepfm, edgecolor="black", linewidth=2)
    ax.set_ylabel("Number of Features", fontsize=12, fontweight="bold")
    ax.set_title("DeepFM+SVD++ Input Features (ITM-Rec)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 150)
    for bar, val in zip(bars, deepfm_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # ITM-Rec DQN
    ax = axes[0, 1]
    dqn_itmrec_categories = [
        "User Emb.\n(32)",
        "Context\n(10)",
        "Demographics\n(6)",
        "History\n(15)",
        "Temporal\n(2)"
    ]
    dqn_itmrec_values = [32, 10, 6, 15, 2]
    colors_dqn_itmrec = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax.bar(dqn_itmrec_categories, dqn_itmrec_values, color=colors_dqn_itmrec, 
                  edgecolor="black", linewidth=2)
    ax.set_ylabel("Number of Components", fontsize=12, fontweight="bold")
    ax.set_title("DQN State Components (ITM-Rec, dim=65)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 35)
    for bar, val in zip(bars, dqn_itmrec_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # OULAD DeepFM
    ax = axes[1, 0]
    deepfm_oulad_categories = ["Embeddings\n(128)", "Context\n(24)", "Linear\n(3)", "FM\n(pairwise)"]
    deepfm_oulad_values = [128, 24, 3, 1]
    colors_deepfm_oulad = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(deepfm_oulad_categories, deepfm_oulad_values, color=colors_deepfm_oulad, 
                  edgecolor="black", linewidth=2)
    ax.set_ylabel("Number of Features", fontsize=12, fontweight="bold")
    ax.set_title("DeepFM+SVD++ Input Features (OULAD)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 150)
    for bar, val in zip(bars, deepfm_oulad_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # OULAD DQN
    ax = axes[1, 1]
    dqn_oulad_categories = [
        "User Emb.\n(32)",
        "Module/Pres\n(20)",
        "Demographics\n(12)",
        "Progress\n(8)",
        "History Agg.\n(16)",
        "Action Avail.\n(4)",
        "Current Proxy\n(4)"
    ]
    dqn_oulad_values = [32, 20, 12, 8, 16, 4, 4]
    colors_dqn_oulad = plt.cm.Set3(np.linspace(0, 1, len(dqn_oulad_values)))
    bars = ax.bar(dqn_oulad_categories, dqn_oulad_values, color=colors_dqn_oulad, 
                  edgecolor="black", linewidth=2)
    ax.set_ylabel("Number of Components", fontsize=12, fontweight="bold")
    ax.set_title("DQN State Components (OULAD, dim=96)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 35)
    for bar, val in zip(bars, dqn_oulad_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    output_file = output_dir / "architecture_comparison_both_datasets.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_comparison_plot():
    """Create direct comparison plot between datasets."""
    print("\nGenerating dataset comparison plot...")
    
    # Load both results
    itmrec_file = output_dir / "dqn_state_importance_itmrec.csv"
    oulad_file = output_dir / "dqn_state_importance_oulad.csv"
    
    if not itmrec_file.exists() or not oulad_file.exists():
        print("  Skipping comparison plot (missing data files)")
        return
    
    itmrec_results = pd.read_csv(itmrec_file)
    oulad_results = pd.read_csv(oulad_file)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("DQN State Importance: ITM-Rec vs OULAD", fontsize=16, fontweight="bold")
    
    # ITM-Rec
    ax = axes[0]
    itmrec_components = itmrec_results["component"].values
    itmrec_importances = itmrec_results["importance"].values
    sorted_idx1 = np.argsort(itmrec_importances)
    colors1 = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(itmrec_components)))
    
    ax.barh(range(len(sorted_idx1)), itmrec_importances[sorted_idx1], 
            color=colors1[sorted_idx1], edgecolor="black", linewidth=1.5)
    ax.set_yticks(range(len(sorted_idx1)))
    ax.set_yticklabels(itmrec_components[sorted_idx1], fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
    ax.set_title("ITM-Rec (state_dim=65)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    
    for i, (idx, val) in enumerate(zip(sorted_idx1, itmrec_importances[sorted_idx1])):
        ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontweight="bold")
    
    # OULAD
    ax = axes[1]
    oulad_components = oulad_results["component"].values
    oulad_importances = oulad_results["importance"].values
    sorted_idx2 = np.argsort(oulad_importances)
    colors2 = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(oulad_components)))
    
    ax.barh(range(len(sorted_idx2)), oulad_importances[sorted_idx2], 
            color=colors2[sorted_idx2], edgecolor="black", linewidth=1.5)
    ax.set_yticks(range(len(sorted_idx2)))
    ax.set_yticklabels(oulad_components[sorted_idx2], fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
    ax.set_title("OULAD (state_dim=96)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    
    for i, (idx, val) in enumerate(zip(sorted_idx2, oulad_importances[sorted_idx2])):
        ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontweight="bold")
    
    plt.tight_layout()
    output_file = output_dir / "dataset_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    print(f"Output directory: {output_dir}\n")
    
    # Generate visualizations for both datasets
    for dataset in ["itmrec", "oulad"]:
        create_importance_visualizations(dataset)
    
    # Create architecture comparison
    create_architecture_comparison_plots()
    
    # Create dataset comparison
    create_comparison_plot()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  + feature_importance_visualization_itmrec.png")
    print("  + feature_importance_visualization_oulad.png")
    print("  + architecture_comparison_both_datasets.png")
    print("  + dataset_comparison.png")
