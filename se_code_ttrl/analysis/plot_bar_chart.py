#!/usr/bin/env python3
"""
Plot bar charts showing evaluation results across different training datasets.

Each subplot: training on one dataset, evaluating on other 5 datasets
Each bar group: Qwen2.5-Math-1.5B (pretrained) vs TTRL (last step) vs TTCS (best iter)
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Set serif font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Times New Roman', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# Configuration
BASE_DIR = "/home/ycy/data1/Self-evolving-Agent/se_code_ttrl/analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "ttcs_vs_ttrl_general_math")

# Colors - consistent with plot_combined_figure.py
COLORS = {
    'TTCS': '#E64B35',       # Red
    'TTRL': '#4DBBD5',       # Cyan/Blue
    'Pretrained': '#BFBFBF', # Light gray
    'R-Zero': '#00A087',     # Teal/Green
}

# Pretrained model results (Qwen2.5-Math-1.5B)
PRETRAINED_RESULTS = {
    "AIME24": 7.1,
    "AIME25": 4.2,
    "AMC23": 27.5,
    "MATH500": 33.2,
    "Minerva": 9.6,
    "OlympiadBench": 22.2,
}

# TTCS training datasets and their directory names
TTCS_DATASETS = {
    "AIME24": "data_AIME24_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step15_temperature0.6",
    "AIME25": "data_AIME25_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step15_temperature0.6",
    "AMC23": "data_AMC23_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
    "MATH500": "data_MATH500_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step15_temperature0.6",
    "Minerva": "data_Minerva_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
    "OlympiadBench": "data_OlympiadBench_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
}

# TTRL training datasets and their directory names
TTRL_DATASETS = {
    "AIME24": "ttrl_Qwen2.5-Math-1.5B_AIME24_bsz8_epoch80_temperature0.6",
    "AIME25": "ttrl_Qwen2.5-Math-1.5B_AIME25_bsz8_epoch80_temperature0.6",
    "AMC23": "ttrl_Qwen2.5-Math-1.5B_AMC23_bsz8_epoch30_temperature0.6",
    "MATH500": "ttrl_Qwen2.5-Math-1.5B_MATH500_bsz32_epoch10_temperature0.6",
    "Minerva": "ttrl_Qwen2.5-Math-1.5B_Minerva_bsz32_epoch10_temperature0.6",
    "OlympiadBench": "ttrl_Qwen2.5-Math-1.5B_OlympiadBench_bsz32_epoch10_temperature0.6",
}

# Datasets that use greedy_data_Overall_results.jsonl with checked_mean@1
GREEDY_DATASETS = ["AMC23", "MATH500", "Minerva", "OlympiadBench"]

# Datasets that use temp_data_Overall_results.jsonl with checked_mean@32
TEMP_DATASETS = ["AIME24", "AIME25"]

ALL_EVAL_DATASETS = list(PRETRAINED_RESULTS.keys())


def get_ttrl_last_step(ttrl_dir):
    """Get the last (largest) step number from TTRL directory."""
    ttrl_path = os.path.join(BASE_DIR, ttrl_dir)
    steps = []
    
    if not os.path.exists(ttrl_path):
        return None
    
    for item in os.listdir(ttrl_path):
        if item.startswith("step_"):
            try:
                step_num = int(item.replace("step_", ""))
                steps.append(step_num)
            except ValueError:
                continue
    
    return max(steps) if steps else None


def get_ttrl_result(ttrl_dir, eval_dataset):
    """
    Get TTRL evaluation result for eval_dataset from the last step.
    """
    last_step = get_ttrl_last_step(ttrl_dir)
    if last_step is None:
        return None
    
    step_path = os.path.join(BASE_DIR, ttrl_dir, f"step_{last_step}")
    
    # Determine which file and field to use
    if eval_dataset in GREEDY_DATASETS:
        file_name = "greedy_data_Overall_results.jsonl"
        field_name = "checked_mean@1"
    else:  # AIME24 or AIME25
        file_name = "temp_data_Overall_results.jsonl"
        field_name = "checked_mean@32"
    
    results_file = os.path.join(step_path, file_name)
    
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get("data_source") == eval_dataset:
                            acc_str = data.get(field_name, "0")
                            return float(acc_str) if acc_str else 0
        except Exception as e:
            print(f"Error reading {results_file}: {e}")
    
    return None


def get_ttcs_best_result(ttcs_dir, eval_dataset):
    """
    Get the best TTCS evaluation result for eval_dataset across all V* iterations.
    """
    ttcs_path = os.path.join(BASE_DIR, ttcs_dir)
    best_result = None
    
    # Determine which file and field to use
    if eval_dataset in GREEDY_DATASETS:
        file_name = "greedy_data_Overall_results.jsonl"
        field_name = "checked_mean@1"
    else:  # AIME24 or AIME25
        file_name = "temp_data_Overall_results.jsonl"
        field_name = "checked_mean@32"
    
    # Find all V* subdirectories
    pattern = os.path.join(ttcs_path, "*-V*")
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


def collect_all_results():
    """
    Collect all evaluation results for TTRL and TTCS.
    Returns: {train_dataset: {eval_dataset: {'ttrl': acc, 'ttcs': acc}}}
    """
    results = {}
    
    for train_name in ALL_EVAL_DATASETS:
        results[train_name] = {}
        ttrl_dir = TTRL_DATASETS[train_name]
        ttcs_dir = TTCS_DATASETS[train_name]
        
        # Get results for each evaluation dataset (excluding the training dataset itself)
        for eval_name in ALL_EVAL_DATASETS:
            if eval_name != train_name:
                ttrl_acc = get_ttrl_result(ttrl_dir, eval_name)
                ttcs_acc = get_ttcs_best_result(ttcs_dir, eval_name)
                results[train_name][eval_name] = {
                    'ttrl': ttrl_acc,
                    'ttcs': ttcs_acc
                }
    
    return results


def plot_single_figure(results, train_datasets_subset, nrows, ncols, output_name):
    """Create a bar chart visualization for a subset of training datasets."""
    from matplotlib.patches import Patch
    
    # Font sizes - consistent with plot_combined_figure.py
    TITLE_SIZE = 18
    LABEL_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 12
    BAR_LABEL_SIZE = 10
    XTICK_SIZE = 11
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5.5*nrows))
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, train_name in enumerate(train_datasets_subset):
        ax = axes[idx]
        
        # Get evaluation datasets (exclude training dataset)
        eval_datasets = [d for d in ALL_EVAL_DATASETS if d != train_name]
        
        # Get pretrained, TTRL, and TTCS results
        pretrained_values = [PRETRAINED_RESULTS[d] for d in eval_datasets]
        ttrl_values = [results[train_name][d]['ttrl'] or 0 for d in eval_datasets]
        ttcs_values = [results[train_name][d]['ttcs'] or 0 for d in eval_datasets]
        
        x = np.arange(len(eval_datasets))
        width = 0.27  # Width for 3 bars
        
        bars1 = ax.bar(x - width, pretrained_values, width, 
                       label='Qwen2.5-Math-1.5B', color=COLORS['Pretrained'])
        bars2 = ax.bar(x, ttrl_values, width, 
                       label='TTRL', color=COLORS['TTRL'])
        bars3 = ax.bar(x + width, ttcs_values, width, 
                       label='TTCS (Ours)', color=COLORS['TTCS'])
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=BAR_LABEL_SIZE)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=BAR_LABEL_SIZE)
        
        for bar in bars3:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=BAR_LABEL_SIZE)
        
        ax.set_xlabel("Evaluation Dataset", fontsize=LABEL_SIZE)
        ax.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
        ax.set_title(f"Trained on {train_name}", fontsize=TITLE_SIZE, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(eval_datasets, rotation=20, ha='right', fontsize=XTICK_SIZE)
        ax.tick_params(axis='y', labelsize=TICK_SIZE)
        ax.grid(True, linestyle='-', alpha=0.6, color='#B0B0B0', linewidth=0.8, axis='y')
        ax.set_axisbelow(True)
    
    # Add unified legend at the top
    legend_elements = [
        Patch(facecolor=COLORS['Pretrained'], label='Qwen2.5-Math-1.5B'),
        Patch(facecolor=COLORS['TTRL'], label='TTRL'),
        Patch(facecolor=COLORS['TTCS'], label='TTCS (Ours)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=LEGEND_SIZE, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(OUTPUT_DIR, f"{output_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    output_pdf = os.path.join(OUTPUT_DIR, f"{output_name}.pdf")
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"PDF saved to: {output_pdf}")
    
    plt.close()


def plot_bar_charts(results):
    """Create bar chart visualizations: 1x3 and 2x3."""
    train_datasets = list(ALL_EVAL_DATASETS)
    
    # Figure 1: 1x3 (first 3 datasets: AIME24, AIME25, AMC23)
    first_3 = train_datasets[:3]
    plot_single_figure(results, first_3, 1, 3, "bar_chart_1x3")
    
    # Figure 2: 2x3 (all 6 datasets)
    plot_single_figure(results, train_datasets, 2, 3, "bar_chart_2x3")


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("\nCollecting evaluation results...")
    results = collect_all_results()
    
    print("\nResults summary:")
    for train_name in ALL_EVAL_DATASETS:
        print(f"\nTrained on {train_name}:")
        for eval_name, accs in results[train_name].items():
            pretrained = PRETRAINED_RESULTS[eval_name]
            ttrl_str = f"{accs['ttrl']:.2f}" if accs['ttrl'] else "N/A"
            ttcs_str = f"{accs['ttcs']:.2f}" if accs['ttcs'] else "N/A"
            print(f"  {eval_name}: Pretrained={pretrained:.1f}, TTRL={ttrl_str}, TTCS={ttcs_str}")
    
    print("\nGenerating bar charts...")
    plot_bar_charts(results)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
