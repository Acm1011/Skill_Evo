#!/usr/bin/env python3
"""
Generate a combined figure with two subfigures:
- Figure (a): TTCS vs TTRL comparison for AIME25 training (line charts, 1x3)
- Figure (b): Bar charts comparing pretrained vs TTCS (1x3)

Both subfigures are generated from evaluation data directly.
"""

import json
import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Set serif font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Times New Roman', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# Configuration
BASE_DIR = "/home/ycy/data1/Self-evolving-Agent/se_code_ttrl/analysis"
PRETRAINED_DIR = os.path.join(BASE_DIR, "Qwen2.5-Math-1.5B")
OUTPUT_DIR = os.path.join(BASE_DIR, "ttcs_vs_ttrl_general_results")

# ============ Configuration for Figure (a): TTCS vs TTRL line charts ============
# For AIME25 training dataset
AIME25_CONFIG = {
    "ttcs_dir": "data_AIME25_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step15_temperature0.6",
    "ttrl_dir": "ttrl_Qwen2.5-Math-1.5B_AIME25_bsz8_epoch80_temperature0.6",
}

# Evaluation datasets for line charts
EVAL_DATASETS_LINE = ["bbeh", "mmlupro", "supergpqa"]

# TTCS iterations to plot
TTCS_ITERS = [1, 5, 10, 15]
X_LABELS_LINE = ["Pretrained\nModel", "Iter 1", "Iter 5", "Iter 10", "Iter 15"]

# ============ Configuration for Figure (b): Bar charts ============
# Pretrained model results
PRETRAINED_RESULTS_BAR = {
    "AIME24": 7.1,
    "AIME25": 4.2,
    "AMC23": 27.5,
    "MATH500": 33.2,
    "Minerva": 9.6,
    "OlympiadBench": 22.2,
}

# Training datasets for bar charts (first 3)
BAR_TRAINING_DATASETS = {
    "AIME24": "data_AIME24_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
    "AIME25": "data_AIME25_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6_backup",
    "AMC23": "data_AMC23_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
}

# Datasets that use different result files
GREEDY_DATASETS = ["AMC23", "MATH500", "Minerva", "OlympiadBench"]
TEMP_DATASETS = ["AIME24", "AIME25"]
ALL_EVAL_DATASETS = ["AIME24", "AIME25", "AMC23", "MATH500", "Minerva", "OlympiadBench"]


# ============ Functions for Figure (a): Line charts ============

def load_pretrained_results_line():
    """Load pretrained results for line chart evaluation datasets."""
    json_path = os.path.join(PRETRAINED_DIR, "aggregated_eval_results.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return {ds: data["datasets"][ds]["accuracy"] for ds in EVAL_DATASETS_LINE}
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load pretrained results: {e}")
        return {ds: None for ds in EVAL_DATASETS_LINE}


def load_ttcs_iter_results(ttcs_dir, iter_num):
    """Load TTCS evaluation results for a specific iteration."""
    base_name = re.sub(r'_step\d+_temperature[\d.]+', '', ttcs_dir)
    iter_dir_name = f"{base_name}-V{iter_num}"
    json_path = os.path.join(BASE_DIR, ttcs_dir, iter_dir_name, "aggregated_eval_results.json")
    
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return {ds: data["datasets"][ds]["accuracy"] for ds in EVAL_DATASETS_LINE}
    except (FileNotFoundError, KeyError):
        return None


def get_ttrl_available_steps(ttrl_dir):
    """Get all available step directories for TTRL."""
    ttrl_path = os.path.join(BASE_DIR, ttrl_dir)
    steps = []
    
    if not os.path.exists(ttrl_path):
        return steps
    
    for item in os.listdir(ttrl_path):
        if item.startswith("step_"):
            step_json = os.path.join(ttrl_path, item, "aggregated_eval_results.json")
            if os.path.exists(step_json):
                step_num = int(item.replace("step_", ""))
                steps.append(step_num)
    
    return sorted(steps)


def map_ttrl_steps_to_iters(steps):
    """Map TTRL steps to iterations."""
    if not steps:
        return {}
    
    mapping = {}
    
    if 20 in steps:
        mapping[1] = 20
        remaining_steps = [s for s in steps if s > 20]
    else:
        mapping[1] = steps[0]
        remaining_steps = steps[1:]
    
    if len(remaining_steps) == 0:
        return mapping
    
    n = len(remaining_steps)
    if n >= 3:
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
        return {ds: data["datasets"][ds]["accuracy"] for ds in EVAL_DATASETS_LINE}
    except (FileNotFoundError, KeyError):
        return None


def collect_line_chart_data():
    """Collect data for AIME25 line charts."""
    ttcs_dir = AIME25_CONFIG["ttcs_dir"]
    ttrl_dir = AIME25_CONFIG["ttrl_dir"]
    
    results = {
        "ttcs": {ds: [] for ds in EVAL_DATASETS_LINE},
        "ttrl": {ds: [] for ds in EVAL_DATASETS_LINE},
    }
    
    # Load pretrained results
    pretrained_results = load_pretrained_results_line()
    for ds in EVAL_DATASETS_LINE:
        results["ttcs"][ds].append(pretrained_results[ds])
        results["ttrl"][ds].append(pretrained_results[ds])
    
    # Load TTCS results
    for iter_num in TTCS_ITERS:
        iter_results = load_ttcs_iter_results(ttcs_dir, iter_num)
        if iter_results:
            for ds in EVAL_DATASETS_LINE:
                results["ttcs"][ds].append(iter_results[ds])
        else:
            for ds in EVAL_DATASETS_LINE:
                results["ttcs"][ds].append(None)
    
    # Load TTRL results
    available_steps = get_ttrl_available_steps(ttrl_dir)
    step_mapping = map_ttrl_steps_to_iters(available_steps)
    print(f"  AIME25 TTRL step mapping: {step_mapping}")
    
    for iter_num in TTCS_ITERS:
        if iter_num in step_mapping:
            step_results = load_ttrl_step_results(ttrl_dir, step_mapping[iter_num])
            if step_results:
                for ds in EVAL_DATASETS_LINE:
                    results["ttrl"][ds].append(step_results[ds])
            else:
                for ds in EVAL_DATASETS_LINE:
                    results["ttrl"][ds].append(None)
        else:
            for ds in EVAL_DATASETS_LINE:
                results["ttrl"][ds].append(None)
    
    return results


# ============ Functions for Figure (b): Bar charts ============

def get_best_result_across_iterations(train_dir, eval_dataset):
    """Get the best evaluation result for eval_dataset across all iterations."""
    train_path = os.path.join(BASE_DIR, train_dir)
    best_result = None
    
    if eval_dataset in GREEDY_DATASETS:
        file_name = "greedy_data_Overall_results.jsonl"
        field_name = "checked_mean@1"
    else:
        file_name = "temp_data_Overall_results.jsonl"
        field_name = "checked_mean@32"
    
    pattern = os.path.join(train_path, "*-V*")
    iter_dirs = glob.glob(pattern)
    
    for iter_dir in iter_dirs:
        results_file = os.path.join(iter_dir, file_name)
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if data.get("data_source") == eval_dataset:
                                acc_str = data.get(field_name, "0")
                                acc = float(acc_str) if acc_str else 0
                                if best_result is None or acc > best_result:
                                    best_result = acc
            except Exception as e:
                print(f"Error reading {results_file}: {e}")
    
    return best_result


def collect_bar_chart_data():
    """Collect data for bar charts."""
    results = {}
    
    for train_name, train_dir in BAR_TRAINING_DATASETS.items():
        results[train_name] = {}
        
        for eval_name in ALL_EVAL_DATASETS:
            if eval_name != train_name:
                best_acc = get_best_result_across_iterations(train_dir, eval_name)
                results[train_name][eval_name] = best_acc
    
    return results


# ============ Combined plotting function ============

def plot_combined_figure(line_data, bar_data):
    """Create a combined figure with subfigures (a) and (b) in one row."""
    
    # Create figure - 1 row x 6 columns, more compact
    fig = plt.figure(figsize=(18, 4))
    
    # Use GridSpec: 1 row, 2 groups (a and b), each with 3 subplots
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.12)
    
    # ============ Figure (a): Line charts (left 3) ============
    gs_a = gs[0].subgridspec(1, 3, wspace=0.35)
    
    colors_line = {
        'TTCS': '#E64B35',
        'TTRL': '#4DBBD5',
    }
    markers_line = {
        'TTCS': 'o',
        'TTRL': 's',
    }
    
    x_line = np.arange(len(X_LABELS_LINE))
    
    for col_idx, eval_ds in enumerate(EVAL_DATASETS_LINE):
        ax = fig.add_subplot(gs_a[col_idx])
        
        # Plot TTCS
        ttcs_values = line_data["ttcs"][eval_ds]
        valid_x_ttcs = [xi for xi, yi in zip(x_line, ttcs_values) if yi is not None]
        valid_y_ttcs = [yi for yi in ttcs_values if yi is not None]
        
        if valid_y_ttcs:
            ax.plot(valid_x_ttcs, valid_y_ttcs, marker=markers_line['TTCS'],
                    color=colors_line['TTCS'], label='TTCS (Ours)',
                    linewidth=2, markersize=7)
        
        # Plot TTRL
        ttrl_values = line_data["ttrl"][eval_ds]
        valid_x_ttrl = [xi for xi, yi in zip(x_line, ttrl_values) if yi is not None]
        valid_y_ttrl = [yi for yi in ttrl_values if yi is not None]
        
        if valid_y_ttrl:
            ax.plot(valid_x_ttrl, valid_y_ttrl, marker=markers_line['TTRL'],
                    color=colors_line['TTRL'], label='TTRL',
                    linewidth=2, markersize=7)
        
        if col_idx == 1:
            ax.set_xlabel("Checkpoint Stage", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(f"{eval_ds.upper()}", fontsize=12, fontweight='bold')
        ax.set_xticks(x_line)
        ax.set_xticklabels(X_LABELS_LINE, rotation=30, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(True, linestyle='-', alpha=0.5, color='#B0B0B0', linewidth=0.6)
        
        if col_idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    # ============ Figure (b): Bar charts (right 3) ============
    gs_b = gs[1].subgridspec(1, 3, wspace=0.35)
    
    color_pretrained = '#E64B35'
    color_best = '#4DBBD5'
    
    train_names_bar = list(BAR_TRAINING_DATASETS.keys())
    
    for idx, train_name in enumerate(train_names_bar):
        ax = fig.add_subplot(gs_b[idx])
        
        eval_datasets = [d for d in ALL_EVAL_DATASETS if d != train_name]
        pretrained_values = [PRETRAINED_RESULTS_BAR[d] for d in eval_datasets]
        best_values = [bar_data[train_name].get(d, 0) or 0 for d in eval_datasets]
        
        x_bar = np.arange(len(eval_datasets))
        width = 0.35
        
        bars1 = ax.bar(x_bar - width/2, pretrained_values, width,
                       label='Qwen2.5-Math-1.5B', color=color_pretrained, alpha=0.85)
        bars2 = ax.bar(x_bar + width/2, best_values, width,
                       label='TTCS', color=color_best, alpha=0.85)
        
        # Add value labels (smaller font)
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 1), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 1), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)
        
        if idx == 1:
            ax.set_xlabel("Evaluation Dataset", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(f"Train: {train_name}", fontsize=12, fontweight='bold')
        ax.set_xticks(x_bar)
        ax.set_xticklabels(eval_datasets, rotation=30, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(True, linestyle='-', alpha=0.5, color='#B0B0B0', linewidth=0.6, axis='y')
        ax.set_axisbelow(True)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=7)
    
    # Add subfigure labels (a) and (b)
    fig.text(0.005, 0.92, '(a)', fontsize=14, fontweight='bold', va='top')
    fig.text(0.505, 0.92, '(b)', fontsize=14, fontweight='bold', va='top')
    
    plt.tight_layout(rect=[0.01, 0, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(OUTPUT_DIR, "combined_figure_ab.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Combined figure saved to: {output_path}")
    
    output_pdf = os.path.join(OUTPUT_DIR, "combined_figure_ab.pdf")
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"Combined PDF saved to: {output_pdf}")
    
    plt.close()


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("\nCollecting data for Figure (a): Line charts...")
    line_data = collect_line_chart_data()
    
    print("\nCollecting data for Figure (b): Bar charts...")
    bar_data = collect_bar_chart_data()
    
    # Print summary
    print("\n" + "="*60)
    print("Data Summary:")
    print("="*60)
    
    print("\nFigure (a) - AIME25 TTCS vs TTRL:")
    for eval_ds in EVAL_DATASETS_LINE:
        ttcs_vals = [f"{v:.2f}" if v else "N/A" for v in line_data["ttcs"][eval_ds]]
        ttrl_vals = [f"{v:.2f}" if v else "N/A" for v in line_data["ttrl"][eval_ds]]
        print(f"  {eval_ds}: TTCS={ttcs_vals}, TTRL={ttrl_vals}")
    
    print("\nFigure (b) - Bar charts:")
    for train_name in BAR_TRAINING_DATASETS:
        print(f"  Trained on {train_name}:")
        for eval_name, acc in bar_data[train_name].items():
            pretrained = PRETRAINED_RESULTS_BAR[eval_name]
            acc_str = f"{acc:.2f}" if acc else "N/A"
            print(f"    {eval_name}: Pretrained={pretrained:.1f}, Best={acc_str}")
    
    print("\nGenerating combined figure...")
    plot_combined_figure(line_data, bar_data)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
