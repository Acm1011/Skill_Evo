#!/usr/bin/env python3
"""
Plot evaluation results across different training datasets and checkpoints.

This script generates a figure with 3 horizontal subplots (bbeh, mmlupro, supergpqa),
each showing the performance of models trained on 6 different datasets across
different checkpoint stages (pretrained, iter 1, 5, 10, 15).
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Set serif font globally (use available alternatives if Times New Roman not installed)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Times New Roman', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # For math text compatibility

# Configuration
BASE_DIR = "/home/ycy/data1/Self-evolving-Agent/se_code_ttrl/analysis"
PRETRAINED_DIR = os.path.join(BASE_DIR, "Qwen2.5-Math-1.5B")

# Colors - consistent with plot_combined_figure.py
COLORS = {
    'TTCS': '#E64B35',       # Red
    'TTRL': '#4DBBD5',       # Cyan/Blue
    'Pretrained': '#BFBFBF', # Light gray
    'R-Zero': '#00A087',     # Teal/Green
}

# R-Zero evaluation results
R_ZERO_RESULTS_PATH = os.path.join(
    BASE_DIR, 
    "qr_R_Zero_gqR_Zero_Qwen2.5-Math-1.5B_step15_temperature0.6",
    "qr_R_Zero_gqR_Zero_Qwen2.5-Math-1.5B-V2",
    "aggregated_eval_results.json"
)

# Training datasets (6 lines in each plot)
TRAINING_DATASETS = {
    "AIME24": "data_AIME24_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
    "AIME25": "data_AIME25_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6_backup",
    "AMC23": "data_AMC23_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
    "MATH500": "data_MATH500_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
    "Minerva": "data_Minerva_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
    "OlympiadBench": "data_OlympiadBench_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
}

# Evaluation datasets (3 subplots)
EVAL_DATASETS = ["bbeh", "mmlupro", "supergpqa"]

# Checkpoint iterations to plot
ITERS = [1, 5, 10, 15]  # V1, V5, V10, V15

# X-axis labels
X_LABELS = ["Pretrained Model"] + [f"Iter {i}" for i in ITERS]


def load_pretrained_results():
    """Load evaluation results for the pretrained model."""
    json_path = os.path.join(PRETRAINED_DIR, "aggregated_eval_results.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    return {ds: data["datasets"][ds]["accuracy"] for ds in EVAL_DATASETS}


def load_r_zero_results():
    """Load R-Zero evaluation results."""
    try:
        with open(R_ZERO_RESULTS_PATH, "r") as f:
            data = json.load(f)
        return {ds: data["datasets"][ds]["accuracy"] for ds in EVAL_DATASETS}
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load R-Zero results: {e}")
        return {ds: None for ds in EVAL_DATASETS}


def load_iter_results(training_dataset_dir, iter_num):
    """Load evaluation results for a specific iteration."""
    # Construct the directory name for this iteration
    base_name = training_dataset_dir.replace("_step10_temperature0.6", "").replace("_backup", "")
    if "_backup" in training_dataset_dir:
        base_name = training_dataset_dir.replace("_step10_temperature0.6_backup", "")
    
    iter_dir_name = f"{base_name}-V{iter_num}"
    json_path = os.path.join(
        BASE_DIR, training_dataset_dir, iter_dir_name, "aggregated_eval_results.json"
    )
    
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return {ds: data["datasets"][ds]["accuracy"] for ds in EVAL_DATASETS}
    except FileNotFoundError:
        print(f"Warning: File not found: {json_path}")
        return None


def collect_all_results():
    """Collect all evaluation results."""
    results = {}
    
    # Load pretrained results (same for all training datasets)
    pretrained_results = load_pretrained_results()
    
    # Load R-Zero results
    r_zero_results = load_r_zero_results()
    results["r_zero"] = r_zero_results
    print(f"R-Zero results: {r_zero_results}")
    
    for train_name, train_dir in TRAINING_DATASETS.items():
        results[train_name] = {ds: [pretrained_results[ds]] for ds in EVAL_DATASETS}
        
        for iter_num in ITERS:
            iter_results = load_iter_results(train_dir, iter_num)
            if iter_results:
                for ds in EVAL_DATASETS:
                    results[train_name][ds].append(iter_results[ds])
            else:
                # If data not found, append None
                for ds in EVAL_DATASETS:
                    results[train_name][ds].append(None)
    
    return results


def plot_results(results):
    """Create the visualization."""
    from matplotlib.lines import Line2D
    
    # Set up the figure with 3 subplots, leave space at top for legend
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Academic-style color palette - muted and professional
    colors = [
        '#E64B35',  # Muted red
        '#4DBBD5',  # Muted cyan
        '#3C5488',  # Muted navy blue
        '#F39B7F',  # Muted peach
        '#8491B4',  # Muted slate blue
        '#7E6148',  # Muted brown
    ]
    
    # Markers for different training datasets
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    x = np.arange(len(X_LABELS))
    
    # Get R-Zero results
    r_zero_results = results.get("r_zero", {})
    
    for ax_idx, eval_ds in enumerate(EVAL_DATASETS):
        ax = axes[ax_idx]
        
        # Plot R-Zero as horizontal dashed line
        r_zero_value = r_zero_results.get(eval_ds)
        if r_zero_value is not None:
            ax.axhline(y=r_zero_value, color=COLORS['R-Zero'], linestyle='--', 
                      linewidth=2.5, label='R-Zero', zorder=1)
        
        for train_idx, (train_name, _) in enumerate(TRAINING_DATASETS.items()):
            y_values = results[train_name][eval_ds]
            
            # Handle None values (missing data)
            valid_x = [xi for xi, yi in zip(x, y_values) if yi is not None]
            valid_y = [yi for yi in y_values if yi is not None]
            
            ax.plot(
                valid_x, valid_y,
                marker=markers[train_idx],
                color=colors[train_idx],
                label=train_name,
                linewidth=2,
                markersize=7,
                zorder=2
            )
        
        ax.set_xlabel("Checkpoint Stage", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"Evaluation on {eval_ds.upper()}", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(X_LABELS, rotation=15)
        ax.grid(True, linestyle='-', alpha=0.6, color='#B0B0B0', linewidth=0.8)
    
    # Create custom legend elements
    legend_elements = [
        Line2D([0], [0], color=COLORS['R-Zero'], linestyle='--', linewidth=2.5, label='R-Zero'),
    ]
    for train_idx, (train_name, _) in enumerate(TRAINING_DATASETS.items()):
        legend_elements.append(
            Line2D([0], [0], color=colors[train_idx], marker=markers[train_idx], 
                   linestyle='-', linewidth=2, markersize=7, label=train_name)
        )
    
    # Add a single shared legend at the top, horizontally arranged
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(TRAINING_DATASETS) + 1,  # +1 for R-Zero
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=False
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space at top for legend
    
    # Save the figure
    output_path = os.path.join(BASE_DIR, "eval_results_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    # Also save as PDF for high quality
    output_pdf = os.path.join(BASE_DIR, "eval_results_comparison.pdf")
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"PDF saved to: {output_pdf}")
    
    plt.show()


def main():
    print("Collecting evaluation results...")
    results = collect_all_results()
    
    print("\nResults summary:")
    
    # Print R-Zero results
    r_zero = results.get("r_zero", {})
    print("\nR-Zero:")
    for eval_ds in EVAL_DATASETS:
        val = r_zero.get(eval_ds)
        print(f"  {eval_ds}: {val:.2f}" if val else f"  {eval_ds}: N/A")
    
    for train_name in TRAINING_DATASETS:
        print(f"\n{train_name}:")
        for eval_ds in EVAL_DATASETS:
            values = results[train_name][eval_ds]
            values_str = [f"{v:.2f}" if v else "N/A" for v in values]
            print(f"  {eval_ds}: {values_str}")
    
    print("\nGenerating plot...")
    plot_results(results)


if __name__ == "__main__":
    main()
