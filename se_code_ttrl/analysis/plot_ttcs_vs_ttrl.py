#!/usr/bin/env python3
"""
Plot comparison between TTCS (iterative loop) and TTRL (continuous training) methods.

This script generates 6 figures (one per training dataset), each with 3 subplots
(bbeh, mmlupro, supergpqa) showing the comparison between TTCS and TTRL methods.
"""

import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Set serif font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Times New Roman', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# Configuration
BASE_DIR = "/home/ycy/data1/Self-evolving-Agent/se_code_ttrl/analysis"
PRETRAINED_DIR = os.path.join(BASE_DIR, "Qwen2.5-Math-1.5B")
OUTPUT_DIR = os.path.join(BASE_DIR, "ttcs_vs_ttrl_general_results")

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

# Training datasets configuration
# TTCS: data_{Dataset}_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step{N}_temperature0.6
# TTRL: ttrl_Qwen2.5-Math-1.5B_{Dataset}_bsz{N}_epoch{M}_temperature0.6
TRAINING_DATASETS = {
    "AIME24": {
        "ttcs_dir": "data_AIME24_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step15_temperature0.6",
        "ttrl_dir": "ttrl_Qwen2.5-Math-1.5B_AIME24_bsz8_epoch80_temperature0.6",
    },
    "AIME25": {
        "ttcs_dir": "data_AIME25_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step15_temperature0.6",
        "ttrl_dir": "ttrl_Qwen2.5-Math-1.5B_AIME25_bsz8_epoch80_temperature0.6",
    },
    "AMC23": {
        "ttcs_dir": "data_AMC23_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
        "ttrl_dir": "ttrl_Qwen2.5-Math-1.5B_AMC23_bsz8_epoch30_temperature0.6",
    },
    "MATH500": {
        "ttcs_dir": "data_MATH500_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step15_temperature0.6",
        "ttrl_dir": "ttrl_Qwen2.5-Math-1.5B_MATH500_bsz32_epoch10_temperature0.6",
    },
    "Minerva": {
        "ttcs_dir": "data_Minerva_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
        "ttrl_dir": "ttrl_Qwen2.5-Math-1.5B_Minerva_bsz32_epoch10_temperature0.6",
    },
    "OlympiadBench": {
        "ttcs_dir": "data_OlympiadBench_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
        "ttrl_dir": "ttrl_Qwen2.5-Math-1.5B_OlympiadBench_bsz32_epoch10_temperature0.6",
    },
}

# Evaluation datasets (3 subplots per figure)
EVAL_DATASETS = ["bbeh", "mmlupro", "supergpqa"]

# TTCS iterations to plot (V1, V5, V10, V15)
TTCS_ITERS = [1, 5, 10, 15]

# X-axis labels
X_LABELS = ["Pretrained Model", "Iter 1", "Iter 5", "Iter 10", "Iter 15"]


def load_pretrained_results():
    """Load evaluation results for the pretrained model."""
    json_path = os.path.join(PRETRAINED_DIR, "aggregated_eval_results.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return {ds: data["datasets"][ds]["accuracy"] for ds in EVAL_DATASETS}
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load pretrained results: {e}")
        return {ds: None for ds in EVAL_DATASETS}


def load_r_zero_results():
    """Load R-Zero evaluation results."""
    try:
        with open(R_ZERO_RESULTS_PATH, "r") as f:
            data = json.load(f)
        return {ds: data["datasets"][ds]["accuracy"] for ds in EVAL_DATASETS}
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load R-Zero results: {e}")
        return {ds: None for ds in EVAL_DATASETS}


def load_ttcs_iter_results(ttcs_dir, iter_num):
    """Load TTCS evaluation results for a specific iteration (V*)."""
    # Extract base name from directory (remove step and temperature suffix)
    base_name = re.sub(r'_step\d+_temperature[\d.]+', '', ttcs_dir)
    iter_dir_name = f"{base_name}-V{iter_num}"
    json_path = os.path.join(BASE_DIR, ttcs_dir, iter_dir_name, "aggregated_eval_results.json")
    
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return {ds: data["datasets"][ds]["accuracy"] for ds in EVAL_DATASETS}
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: TTCS file not found: {json_path}")
        return None


def get_ttrl_available_steps(ttrl_dir):
    """Get all available step directories for TTRL, sorted by step number."""
    ttrl_path = os.path.join(BASE_DIR, ttrl_dir)
    steps = []
    
    if not os.path.exists(ttrl_path):
        return steps
    
    for item in os.listdir(ttrl_path):
        if item.startswith("step_"):
            # Check if aggregated_eval_results.json exists in the step directory
            step_json = os.path.join(ttrl_path, item, "aggregated_eval_results.json")
            if os.path.exists(step_json):
                step_num = int(item.replace("step_", ""))
                steps.append(step_num)
    
    return sorted(steps)


def map_ttrl_steps_to_iters(steps):
    """
    Map TTRL steps to iterations.
    - step20 -> iter1
    - Divide remaining steps into 3 equal intervals for iter5, iter10, iter15
    
    Returns: dict mapping iter number to step number
    """
    if not steps:
        return {}
    
    # iter1 corresponds to step20
    mapping = {}
    
    if 20 in steps:
        mapping[1] = 20
        remaining_steps = [s for s in steps if s > 20]
    else:
        # Use the smallest step as iter1
        mapping[1] = steps[0]
        remaining_steps = steps[1:]
    
    if len(remaining_steps) == 0:
        return mapping
    
    # Divide remaining steps into 3 roughly equal parts for iter5, iter10, iter15
    n = len(remaining_steps)
    if n >= 3:
        # Pick steps at 1/3, 2/3, and end positions
        indices = [n // 3 - 1, 2 * n // 3 - 1, n - 1]
        for i, iter_num in enumerate([5, 10, 15]):
            idx = max(0, min(indices[i], n - 1))
            mapping[iter_num] = remaining_steps[idx]
    elif n == 2:
        mapping[5] = remaining_steps[0]
        mapping[15] = remaining_steps[1]
    elif n == 1:
        mapping[15] = remaining_steps[0]
    
    return mapping


def load_ttrl_step_results(ttrl_dir, step_num):
    """Load TTRL evaluation results for a specific step."""
    json_path = os.path.join(BASE_DIR, ttrl_dir, f"step_{step_num}", "aggregated_eval_results.json")
    
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return {ds: data["datasets"][ds]["accuracy"] for ds in EVAL_DATASETS}
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: TTRL file not found: {json_path}")
        return None


def collect_results_for_training_dataset(train_name, config):
    """Collect all results (TTCS and TTRL) for one training dataset."""
    ttcs_dir = config["ttcs_dir"]
    ttrl_dir = config["ttrl_dir"]
    
    # Initialize results
    results = {
        "ttcs": {ds: [] for ds in EVAL_DATASETS},
        "ttrl": {ds: [] for ds in EVAL_DATASETS},
        "ttrl_step_mapping": {}
    }
    
    # Load pretrained results (same for both methods)
    pretrained_results = load_pretrained_results()
    for ds in EVAL_DATASETS:
        results["ttcs"][ds].append(pretrained_results[ds])
        results["ttrl"][ds].append(pretrained_results[ds])
    
    # Load TTCS results for each iteration
    for iter_num in TTCS_ITERS:
        iter_results = load_ttcs_iter_results(ttcs_dir, iter_num)
        if iter_results:
            for ds in EVAL_DATASETS:
                results["ttcs"][ds].append(iter_results[ds])
        else:
            for ds in EVAL_DATASETS:
                results["ttcs"][ds].append(None)
    
    # Load TTRL results - first get available steps and create mapping
    available_steps = get_ttrl_available_steps(ttrl_dir)
    step_mapping = map_ttrl_steps_to_iters(available_steps)
    results["ttrl_step_mapping"] = step_mapping
    
    print(f"  {train_name} TTRL step mapping: {step_mapping}")
    
    # Load TTRL results for each mapped iteration
    for iter_num in TTCS_ITERS:
        if iter_num in step_mapping:
            step_results = load_ttrl_step_results(ttrl_dir, step_mapping[iter_num])
            if step_results:
                for ds in EVAL_DATASETS:
                    results["ttrl"][ds].append(step_results[ds])
            else:
                for ds in EVAL_DATASETS:
                    results["ttrl"][ds].append(None)
        else:
            for ds in EVAL_DATASETS:
                results["ttrl"][ds].append(None)
    
    return results


def plot_comparison(all_results, r_zero_results):
    """Create 6 figures (one per training dataset), each with 3 subplots."""
    from matplotlib.lines import Line2D
    
    markers = {
        'TTCS': 'o',
        'TTRL': 's',
    }
    
    x = np.arange(len(X_LABELS))
    
    for train_name, results in all_results.items():
        # Create a figure with 3 subplots (one for each eval dataset)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for ax_idx, eval_ds in enumerate(EVAL_DATASETS):
            ax = axes[ax_idx]
            
            # Plot R-Zero as horizontal dashed line
            r_zero_value = r_zero_results.get(eval_ds)
            if r_zero_value is not None:
                ax.axhline(y=r_zero_value, color=COLORS['R-Zero'], linestyle='--', 
                          linewidth=2.5, label='R-Zero', zorder=1)
            
            # Plot TTCS
            ttcs_values = results["ttcs"][eval_ds]
            valid_x_ttcs = [xi for xi, yi in zip(x, ttcs_values) if yi is not None]
            valid_y_ttcs = [yi for yi in ttcs_values if yi is not None]
            
            if valid_y_ttcs:
                ax.plot(
                    valid_x_ttcs, valid_y_ttcs,
                    marker=markers['TTCS'],
                    color=COLORS['TTCS'],
                    label='TTCS (Ours)',
                    linewidth=2,
                    markersize=8,
                    zorder=2
                )
            
            # Plot TTRL
            ttrl_values = results["ttrl"][eval_ds]
            valid_x_ttrl = [xi for xi, yi in zip(x, ttrl_values) if yi is not None]
            valid_y_ttrl = [yi for yi in ttrl_values if yi is not None]
            
            if valid_y_ttrl:
                ax.plot(
                    valid_x_ttrl, valid_y_ttrl,
                    marker=markers['TTRL'],
                    color=COLORS['TTRL'],
                    label='TTRL',
                    linewidth=2,
                    markersize=8,
                    zorder=2
                )
            
            ax.set_xlabel("Checkpoint Stage", fontsize=14)
            ax.set_ylabel("Accuracy (%)", fontsize=14)
            ax.set_title(f"Evaluation on {eval_ds.upper()}", fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(X_LABELS, rotation=15, ha='right', fontsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.grid(True, linestyle='-', alpha=0.6, color='#B0B0B0', linewidth=0.8)
        
        # Create unified legend
        legend_elements = [
            Line2D([0], [0], color=COLORS['R-Zero'], linestyle='--', linewidth=2.5, label='R-Zero'),
            Line2D([0], [0], color=COLORS['TTRL'], marker='s', linestyle='-', linewidth=2, markersize=8, label='TTRL'),
            Line2D([0], [0], color=COLORS['TTCS'], marker='o', linestyle='-', linewidth=2, markersize=8, label='TTCS (Ours)'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=3, fontsize=12, frameon=False)
        
        # Add overall title
        fig.suptitle(f"Training Dataset: {train_name}", fontsize=18, fontweight='bold', y=1.08)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        output_path = os.path.join(OUTPUT_DIR, f"ttcs_vs_ttrl_{train_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
        
        output_pdf = os.path.join(OUTPUT_DIR, f"ttcs_vs_ttrl_{train_name}.pdf")
        plt.savefig(output_pdf, bbox_inches='tight')
        print(f"PDF saved to: {output_pdf}")
        
        plt.close()


def create_combined_figure(all_results, r_zero_results):
    """Create a single combined figure with 6x3 layout (6 training datasets x 3 eval datasets)."""
    from matplotlib.lines import Line2D
    
    markers = {
        'TTCS': 'o',
        'TTRL': 's',
    }
    
    x = np.arange(len(X_LABELS))
    train_names = list(all_results.keys())
    
    # Create 6x3 grid: rows = training datasets, columns = evaluation datasets
    fig, axes = plt.subplots(6, 3, figsize=(15, 26))
    
    for row_idx, train_name in enumerate(train_names):
        results = all_results[train_name]
        
        for col_idx, eval_ds in enumerate(EVAL_DATASETS):
            ax = axes[row_idx, col_idx]
            
            # Plot R-Zero as horizontal dashed line
            r_zero_value = r_zero_results.get(eval_ds)
            if r_zero_value is not None:
                ax.axhline(y=r_zero_value, color=COLORS['R-Zero'], linestyle='--', 
                          linewidth=2.5, zorder=1)
            
            # Plot TTCS
            ttcs_values = results["ttcs"][eval_ds]
            valid_x_ttcs = [xi for xi, yi in zip(x, ttcs_values) if yi is not None]
            valid_y_ttcs = [yi for yi in ttcs_values if yi is not None]
            
            if valid_y_ttcs:
                ax.plot(
                    valid_x_ttcs, valid_y_ttcs,
                    marker=markers['TTCS'],
                    color=COLORS['TTCS'],
                    linewidth=2.5,
                    markersize=8,
                    zorder=2
                )
            
            # Plot TTRL
            ttrl_values = results["ttrl"][eval_ds]
            valid_x_ttrl = [xi for xi, yi in zip(x, ttrl_values) if yi is not None]
            valid_y_ttrl = [yi for yi in ttrl_values if yi is not None]
            
            if valid_y_ttrl:
                ax.plot(
                    valid_x_ttrl, valid_y_ttrl,
                    marker=markers['TTRL'],
                    color=COLORS['TTRL'],
                    linewidth=2.5,
                    markersize=8,
                    zorder=2
                )
            
            # Set labels
            if row_idx == 5:  # Bottom row
                ax.set_xlabel("Checkpoint Stage", fontsize=14)
            if col_idx == 0:  # Left column
                ax.set_ylabel("Accuracy (%)", fontsize=14)
            
            # Title: show training dataset on left column, eval dataset on top row
            if row_idx == 0:
                ax.set_title(f"Eval: {eval_ds.upper()}", fontsize=16, fontweight='bold')
            
            # Add training dataset label on the right side of the rightmost column
            if col_idx == 2:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"Train: {train_name}", fontsize=14, fontweight='bold', 
                             rotation=270, labelpad=25)
            
            ax.set_xticks(x)
            ax.set_xticklabels(X_LABELS, rotation=25, ha='right', fontsize=11)
            ax.tick_params(axis='y', labelsize=11)
            ax.grid(True, linestyle='-', alpha=0.6, color='#B0B0B0', linewidth=0.8)
    
    # Create unified legend at the top
    legend_elements = [
        Line2D([0], [0], color=COLORS['R-Zero'], linestyle='--', linewidth=2.5, label='R-Zero'),
        Line2D([0], [0], color=COLORS['TTRL'], marker='s', linestyle='-', linewidth=2.5, markersize=8, label='TTRL'),
        Line2D([0], [0], color=COLORS['TTCS'], marker='o', linestyle='-', linewidth=2.5, markersize=8, label='TTCS (Ours)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.995),
               ncol=3, fontsize=14, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save combined figure
    output_path = os.path.join(OUTPUT_DIR, "ttcs_vs_ttrl_combined.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Combined figure saved to: {output_path}")
    
    output_pdf = os.path.join(OUTPUT_DIR, "ttcs_vs_ttrl_combined.pdf")
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Combined PDF saved to: {output_pdf}")
    
    plt.close()


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("Collecting evaluation results...")
    all_results = {}
    
    # Load R-Zero results
    r_zero_results = load_r_zero_results()
    print(f"\nR-Zero results: {r_zero_results}")
    
    for train_name, config in TRAINING_DATASETS.items():
        print(f"\nProcessing {train_name}...")
        all_results[train_name] = collect_results_for_training_dataset(train_name, config)
    
    # Print summary
    print("\n" + "="*80)
    print("Results Summary:")
    print("="*80)
    
    print("\nR-Zero:")
    for eval_ds in EVAL_DATASETS:
        val = r_zero_results.get(eval_ds)
        print(f"  {eval_ds}: {val:.2f}" if val else f"  {eval_ds}: N/A")
    
    for train_name, results in all_results.items():
        print(f"\n{train_name}:")
        print(f"  TTRL step mapping: {results['ttrl_step_mapping']}")
        for eval_ds in EVAL_DATASETS:
            ttcs_values = [f"{v:.2f}" if v else "N/A" for v in results["ttcs"][eval_ds]]
            ttrl_values = [f"{v:.2f}" if v else "N/A" for v in results["ttrl"][eval_ds]]
            print(f"  {eval_ds}:")
            print(f"    TTCS: {ttcs_values}")
            print(f"    TTRL: {ttrl_values}")
    
    print("\nGenerating plots...")
    plot_comparison(all_results, r_zero_results)
    create_combined_figure(all_results, r_zero_results)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
