#!/usr/bin/env python3
"""
Combined figure with:
- Left (a): TTCS vs TTRL line charts for AIME25 training (1x3)
- Right (b): Bar charts comparing Qwen2.5-Math-1.5B, TTRL, TTCS (1x3)
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
OUTPUT_DIR = os.path.join(BASE_DIR, "combined")

# Colors - consistent across both plots
COLORS = {
    'TTCS': '#E64B35',   # Red
    'TTRL': '#4DBBD5',   # Cyan/Blue
    'Pretrained': '#BFBFBF',  # Light gray
    'R-Zero': '#00A087',  # Teal/Green
}

# R-Zero evaluation results
R_ZERO_RESULTS_PATH = os.path.join(
    BASE_DIR, 
    "qr_R_Zero_gqR_Zero_Qwen2.5-Math-1.5B_step15_temperature0.6",
    "qr_R_Zero_gqR_Zero_Qwen2.5-Math-1.5B-V2",
    "aggregated_eval_results.json"
)

# ==================== Line chart configuration (Left side) ====================
AIME25_CONFIG = {
    "ttcs_dir": "data_AIME25_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step15_temperature0.6",
    "ttrl_dir": "ttrl_Qwen2.5-Math-1.5B_AIME25_bsz8_epoch80_temperature0.6",
}

EVAL_DATASETS_LINE = ["bbeh", "mmlupro", "supergpqa"]
EVAL_DISPLAY_NAMES = {
    "bbeh": "BBEH",
    "mmlupro": "MMLU-Pro",
    "supergpqa": "SuperGPQA",
}
TTCS_ITERS = [1, 5, 10, 15]
X_LABELS_LINE = ["Pretrained\nModel", "Iter 1", "Iter 5", "Iter 10", "Iter 15"]

# ==================== Bar chart configuration (Right side) ====================
PRETRAINED_RESULTS = {
    "AIME24": 7.1,
    "AIME25": 4.2,
    "AMC23": 27.5,
    "MATH500": 33.2,
    "Minerva": 9.6,
    "OlympiadBench": 22.2,
}

TTCS_DATASETS = {
    "AIME24": "data_AIME24_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
    "AIME25": "data_AIME25_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step15_temperature0.6",
    "AMC23": "data_AMC23_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6",
    "MATH500":"data_MATH500_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step15_temperature0.6"
}

TTRL_DATASETS = {
    "AIME24": "ttrl_Qwen2.5-Math-1.5B_AIME24_bsz8_epoch80_temperature0.6",
    "AIME25": "ttrl_Qwen2.5-Math-1.5B_AIME25_bsz8_epoch80_temperature0.6",
    "AMC23": "ttrl_Qwen2.5-Math-1.5B_AMC23_bsz8_epoch30_temperature0.6",
    'MATH500': "ttrl_Qwen2.5-Math-1.5B_MATH500_bsz32_epoch10_temperature0.6"
}

GREEDY_DATASETS = ["AMC23", "MATH500", "Minerva", "OlympiadBench"]
ALL_EVAL_DATASETS = list(PRETRAINED_RESULTS.keys())


# ==================== Line chart functions ====================

def load_r_zero_results():
    """Load R-Zero evaluation results for line chart."""
    try:
        with open(R_ZERO_RESULTS_PATH, "r") as f:
            data = json.load(f)
        return {ds: data["datasets"][ds]["accuracy"] for ds in EVAL_DATASETS_LINE}
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load R-Zero results: {e}")
        return {ds: None for ds in EVAL_DATASETS_LINE}


def load_pretrained_results_line():
    """Load pretrained results for line chart."""
    json_path = os.path.join(PRETRAINED_DIR, "aggregated_eval_results.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return {ds: data["datasets"][ds]["accuracy"] for ds in EVAL_DATASETS_LINE}
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load pretrained results: {e}")
        return {ds: None for ds in EVAL_DATASETS_LINE}


def load_ttcs_iter_results_line(ttcs_dir, iter_num):
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


def load_ttrl_step_results_line(ttrl_dir, step_num):
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
        "r_zero": {},  # R-Zero results (single value per dataset)
    }
    
    # Load pretrained results
    pretrained_results = load_pretrained_results_line()
    for ds in EVAL_DATASETS_LINE:
        results["ttcs"][ds].append(pretrained_results[ds])
        results["ttrl"][ds].append(pretrained_results[ds])
    
    # Load R-Zero results
    r_zero_results = load_r_zero_results()
    results["r_zero"] = r_zero_results
    print(f"  R-Zero results: {r_zero_results}")
    
    # Load TTCS results
    for iter_num in TTCS_ITERS:
        iter_results = load_ttcs_iter_results_line(ttcs_dir, iter_num)
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
            step_results = load_ttrl_step_results_line(ttrl_dir, step_mapping[iter_num])
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


# ==================== Bar chart functions ====================

def get_ttrl_last_step(ttrl_dir):
    """Get the last step number from TTRL directory."""
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


def get_ttrl_result_bar(ttrl_dir, eval_dataset):
    """Get TTRL evaluation result from the last step."""
    last_step = get_ttrl_last_step(ttrl_dir)
    if last_step is None:
        return None
    
    step_path = os.path.join(BASE_DIR, ttrl_dir, f"step_{last_step}")
    
    if eval_dataset in GREEDY_DATASETS:
        file_name = "greedy_data_Overall_results.jsonl"
        field_name = "checked_mean@1"
    else:
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


def get_ttcs_best_result_bar(ttcs_dir, eval_dataset):
    """Get the best TTCS evaluation result across all V* iterations."""
    ttcs_path = os.path.join(BASE_DIR, ttcs_dir)
    best_result = None
    
    if eval_dataset in GREEDY_DATASETS:
        file_name = "greedy_data_Overall_results.jsonl"
        field_name = "checked_mean@1"
    else:
        file_name = "temp_data_Overall_results.jsonl"
        field_name = "checked_mean@32"
    
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


def collect_bar_chart_data():
    """Collect data for bar charts (first 3 training datasets)."""
    results = {}
    
    for train_name in ["MATH500", "AIME25", "AMC23"]:
        results[train_name] = {}
        ttrl_dir = TTRL_DATASETS[train_name]
        ttcs_dir = TTCS_DATASETS[train_name]
        
        for eval_name in ALL_EVAL_DATASETS:
            if eval_name != train_name:
                ttrl_acc = get_ttrl_result_bar(ttrl_dir, eval_name)
                ttcs_acc = get_ttcs_best_result_bar(ttcs_dir, eval_name)
                results[train_name][eval_name] = {
                    'ttrl': ttrl_acc,
                    'ttcs': ttcs_acc
                }
    
    return results


# ==================== Combined plotting ====================

def plot_combined_figure(line_data, bar_data):
    """Create combined figure with line charts (left) and bar charts (right)."""
    
    # Layout: 3 line charts (left, square) + 3 bar charts (right, wider)
    fig_width, fig_height = 26, 6
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Unified bottom for alignment - more space for rotated x labels and caption
    plot_bottom = 0.36
    plot_top = 0.82
    plot_height = plot_top - plot_bottom

    # Font sizes - follow the 2x2 layout style
    TITLE_SIZE = 18
    LABEL_SIZE = 20
    TICK_SIZE = 16
    LEGEND_SIZE = 18
    BAR_LABEL_SIZE = 10
    XTICK_SIZE = 15

    # Left side: make subplots square
    left_margin = 0.005
    left_subplot_width = plot_height * (fig_height / fig_width)  # Square aspect
    left_gap = 0.02
    
    # Right side: remaining space - wider for bar charts
    gap_between = 0.035
    right_margin = 0.005
    left_end = left_margin + 3 * left_subplot_width + 2 * left_gap
    right_start = left_end + gap_between
    right_available = 1 - right_start - right_margin
    right_gap = 0.018
    right_subplot_width = (right_available - 2 * right_gap) / 3
    
    # ==================== Left side: Line charts (1x3) ====================
    x_line = np.arange(len(X_LABELS_LINE))
    
    for col_idx, eval_ds in enumerate(EVAL_DATASETS_LINE):
        left_pos = left_margin + col_idx * (left_subplot_width + left_gap)
        ax = fig.add_axes([left_pos, plot_bottom, left_subplot_width, plot_height])
        
        # Plot R-Zero as horizontal dashed line
        r_zero_value = line_data["r_zero"].get(eval_ds)
        if r_zero_value is not None:
            ax.axhline(y=r_zero_value, color=COLORS['R-Zero'], linestyle='--', 
                      linewidth=3, label='R-Zero')
        
        # Plot TTCS
        ttcs_values = line_data["ttcs"][eval_ds]
        valid_x_ttcs = [xi for xi, yi in zip(x_line, ttcs_values) if yi is not None]
        valid_y_ttcs = [yi for yi in ttcs_values if yi is not None]
        
        if valid_y_ttcs:
            ax.plot(valid_x_ttcs, valid_y_ttcs, marker='o',
                    color=COLORS['TTCS'], label='TTCS (Ours)',
                    linewidth=3, markersize=10)
        
        # Plot TTRL
        ttrl_values = line_data["ttrl"][eval_ds]
        valid_x_ttrl = [xi for xi, yi in zip(x_line, ttrl_values) if yi is not None]
        valid_y_ttrl = [yi for yi in ttrl_values if yi is not None]
        
        if valid_y_ttrl:
            ax.plot(valid_x_ttrl, valid_y_ttrl, marker='s',
                    color=COLORS['TTRL'], label='TTRL',
                    linewidth=3, markersize=10)
        
        if col_idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
        ax.set_title(EVAL_DISPLAY_NAMES.get(eval_ds, eval_ds.upper()), fontsize=TITLE_SIZE, pad=10)
        ax.set_xticks(x_line)
        ax.set_xticklabels(X_LABELS_LINE, fontsize=XTICK_SIZE, rotation=20, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='y', labelsize=TICK_SIZE)
        ax.grid(True, linestyle='-', alpha=0.6, color='#B0B0B0', linewidth=0.8)
    
    # Add centered legend for left section (order: R-Zero, TTRL, TTCS)
    left_center = left_margin + (3 * left_subplot_width + 2 * left_gap) / 2
    from matplotlib.lines import Line2D
    legend_elements_line = [
        Line2D([0], [0], color=COLORS['R-Zero'], linestyle='--', linewidth=3, label='R-Zero'),
        Line2D([0], [0], color=COLORS['TTRL'], marker='s', linestyle='-', linewidth=3, markersize=12, label='TTRL'),
        Line2D([0], [0], color=COLORS['TTCS'], marker='o', linestyle='-', linewidth=3, markersize=12, label='TTCS (Ours)'),
    ]
    fig.legend(handles=legend_elements_line, loc='upper center', bbox_to_anchor=(left_center, 0.97), 
               ncol=3, fontsize=LEGEND_SIZE, frameon=False)
    
    # ==================== Right side: Bar charts (1x3) ====================
    train_names_bar = ["MATH500", "AIME25", "AMC23"]  # All 3 bar charts
    
    for idx, train_name in enumerate(train_names_bar):
        right_pos = right_start + idx * (right_subplot_width + right_gap)
        ax = fig.add_axes([right_pos, plot_bottom, right_subplot_width, plot_height])
        
        eval_datasets = [d for d in ALL_EVAL_DATASETS if d != train_name]
        pretrained_values = [PRETRAINED_RESULTS[d] for d in eval_datasets]
        ttrl_values = [bar_data[train_name][d]['ttrl'] or 0 for d in eval_datasets]
        ttcs_values = [bar_data[train_name][d]['ttcs'] or 0 for d in eval_datasets]
        
        x_bar = np.arange(len(eval_datasets))
        width = 0.32  # Bar width
        
        bars1 = ax.bar(x_bar - width, pretrained_values, width, 
                       label='Qwen2.5-Math-1.5B', color=COLORS['Pretrained'])
        bars2 = ax.bar(x_bar, ttrl_values, width, 
                       label='TTRL', color=COLORS['TTRL'])
        bars3 = ax.bar(x_bar + width, ttcs_values, width, 
                       label='TTCS (Ours)', color=COLORS['TTCS'])
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2), textcoords="offset points",
                           ha='center', va='bottom', fontsize=BAR_LABEL_SIZE)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2), textcoords="offset points",
                           ha='center', va='bottom', fontsize=BAR_LABEL_SIZE)
        
        for bar in bars3:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2), textcoords="offset points",
                           ha='center', va='bottom', fontsize=BAR_LABEL_SIZE)
        
        ax.set_ylim(0, 80)
        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
        ax.set_title(f"Trained on {train_name}", fontsize=TITLE_SIZE, pad=10)
        ax.set_xticks(x_bar)
        ax.set_xticklabels(eval_datasets, fontsize=XTICK_SIZE, rotation=25, ha='right')
        ax.tick_params(axis='y', labelsize=TICK_SIZE)
        ax.grid(True, linestyle='-', alpha=0.6, color='#B0B0B0', linewidth=0.8, axis='y')
        ax.set_axisbelow(True)
    
    # Add centered legend for right section
    right_center = right_start + right_available / 2
    from matplotlib.patches import Patch
    legend_elements_bar = [
        Patch(facecolor=COLORS['Pretrained'], label='Qwen2.5-Math-1.5B'),
        Patch(facecolor=COLORS['TTRL'], label='TTRL'),
        Patch(facecolor=COLORS['TTCS'], label='TTCS (Ours)'),
    ]
    fig.legend(handles=legend_elements_bar, loc='upper center', bbox_to_anchor=(right_center, 0.97), 
               ncol=3, fontsize=LEGEND_SIZE, frameon=False)
    
    # Add subfigure captions - positioned below x-axis labels
    caption_y = 0.11
    caption_y2 = 0.18

    fig.text(left_center, caption_y, '(a) General-domain performance comparison', 
             fontsize=LABEL_SIZE, ha='center', va='top')
    fig.text(left_center, caption_y2, 'Checkpoint Stage', 
             fontsize=LABEL_SIZE, ha='center', va='top')

    fig.text(right_center, caption_y, '(b) Mathematical-domain performance comparison', 
             fontsize=LABEL_SIZE, ha='center', va='top')
    fig.text(right_center, caption_y2, 'Evaluation Dataset', 
             fontsize=LABEL_SIZE, ha='center', va='top')
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, "combined_figure.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    output_pdf = os.path.join(OUTPUT_DIR, "combined_figure.pdf")
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {output_pdf}")
    
    plt.close()


def plot_combined_figure_2x2(line_data, bar_data):
    """Create combined figure with 2 line charts (left) and 2 bar charts (right)."""
    
    # Layout: 2 line charts (left, square) + 2 bar charts (right, wider)
    # figsize (16, 5) -> scaled to 7" in paper -> scale factor ~0.44 -> 22pt becomes ~10pt
    fig_width, fig_height = 18, 6
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Unified bottom for alignment - more space for rotated x labels and caption
    plot_bottom = 0.36
    plot_top = 0.82
    plot_height = plot_top - plot_bottom

    # Font sizes - will appear as ~10pt when scaled to paper
    TITLE_SIZE = 18
    LABEL_SIZE = 20
    TICK_SIZE = 16
    LEGEND_SIZE = 18
    BAR_LABEL_SIZE = 10
    XTICK_SIZE = 15
    
    # Left side: make subplots square
    left_margin = 0.005
    left_subplot_width = plot_height * 5.3 / 16  # Square aspect
    left_gap = 0.03
    
    # Right side: remaining space - wider for bar charts
    gap_between = 0.05
    right_margin = 0.005
    left_end = left_margin + 2 * left_subplot_width + left_gap
    right_start = left_end + gap_between
    right_available = 1 - right_start - right_margin
    right_gap = 0.03
    right_subplot_width = (right_available - right_gap) / 2
    
    # ==================== Left side: Line charts (1x2) ====================
    x_line = np.arange(len(X_LABELS_LINE))
    eval_datasets_2 = ["mmlupro", "supergpqa"]  # Only 2 datasets
    
    for col_idx, eval_ds in enumerate(eval_datasets_2):
        left_pos = left_margin + col_idx * (left_subplot_width + left_gap)
        ax = fig.add_axes([left_pos, plot_bottom, left_subplot_width, plot_height])
        
        # Plot R-Zero as horizontal dashed line
        r_zero_value = line_data["r_zero"].get(eval_ds)
        if r_zero_value is not None:
            ax.axhline(y=r_zero_value, color=COLORS['R-Zero'], linestyle='--', 
                      linewidth=3, label='R-Zero')
        
        # Plot TTCS
        ttcs_values = line_data["ttcs"][eval_ds]
        valid_x_ttcs = [xi for xi, yi in zip(x_line, ttcs_values) if yi is not None]
        valid_y_ttcs = [yi for yi in ttcs_values if yi is not None]
        
        if valid_y_ttcs:
            ax.plot(valid_x_ttcs, valid_y_ttcs, marker='o',
                    color=COLORS['TTCS'], label='TTCS (Ours)',
                    linewidth=3, markersize=10)
        
        # Plot TTRL
        ttrl_values = line_data["ttrl"][eval_ds]
        valid_x_ttrl = [xi for xi, yi in zip(x_line, ttrl_values) if yi is not None]
        valid_y_ttrl = [yi for yi in ttrl_values if yi is not None]
        
        if valid_y_ttrl:
            ax.plot(valid_x_ttrl, valid_y_ttrl, marker='s',
                    color=COLORS['TTRL'], label='TTRL',
                    linewidth=3, markersize=10)
        
        if col_idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
        ax.set_title(EVAL_DISPLAY_NAMES.get(eval_ds, eval_ds.upper()), fontsize=TITLE_SIZE, pad=10)
        ax.set_xticks(x_line)
        ax.set_xticklabels(X_LABELS_LINE, fontsize=XTICK_SIZE, rotation=20, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='y', labelsize=TICK_SIZE)
        ax.grid(True, linestyle='-', alpha=0.6, color='#B0B0B0', linewidth=0.8)
    
    # Add centered legend for left section (order: R-Zero, TTRL, TTCS)
    left_center = left_margin + left_subplot_width + left_gap / 2
    from matplotlib.lines import Line2D
    legend_elements_line = [
        Line2D([0], [0], color=COLORS['R-Zero'], linestyle='--', linewidth=3, label='R-Zero'),
        Line2D([0], [0], color=COLORS['TTRL'], marker='s', linestyle='-', linewidth=3, markersize=12, label='TTRL'),
        Line2D([0], [0], color=COLORS['TTCS'], marker='o', linestyle='-', linewidth=3, markersize=12, label='TTCS (Ours)'),
    ]
    fig.legend(handles=legend_elements_line, loc='upper center', bbox_to_anchor=(left_center, 0.97), 
               ncol=3, fontsize=LEGEND_SIZE, frameon=False)
    
    # ==================== Right side: Bar charts (1x2) ====================
    train_names_bar = ["AIME25", "MATH500"]  # Only 2 bar charts
    
    for idx, train_name in enumerate(train_names_bar):
        right_pos = right_start + idx * (right_subplot_width + right_gap)
        ax = fig.add_axes([right_pos, plot_bottom, right_subplot_width, plot_height])
        
        eval_datasets = [d for d in ALL_EVAL_DATASETS if d != train_name]
        pretrained_values = [PRETRAINED_RESULTS[d] for d in eval_datasets]
        ttrl_values = [bar_data[train_name][d]['ttrl'] or 0 for d in eval_datasets]
        ttcs_values = [bar_data[train_name][d]['ttcs'] or 0 for d in eval_datasets]
        
        x_bar = np.arange(len(eval_datasets))
        width = 0.32  # Bar width
        
        bars1 = ax.bar(x_bar - width, pretrained_values, width, 
                       label='Qwen2.5-Math-1.5B', color=COLORS['Pretrained'])
        bars2 = ax.bar(x_bar, ttrl_values, width, 
                       label='TTRL', color=COLORS['TTRL'])
        bars3 = ax.bar(x_bar + width, ttcs_values, width, 
                       label='TTCS (Ours)', color=COLORS['TTCS'])
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2), textcoords="offset points",
                           ha='center', va='bottom', fontsize=BAR_LABEL_SIZE)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2), textcoords="offset points",
                           ha='center', va='bottom', fontsize=BAR_LABEL_SIZE)
        
        for bar in bars3:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2), textcoords="offset points",
                           ha='center', va='bottom', fontsize=BAR_LABEL_SIZE)
        
        ax.set_ylim(0, 80)
        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
        ax.set_title(f"Trained on {train_name}", fontsize=TITLE_SIZE, pad=10)
        ax.set_xticks(x_bar)
        ax.set_xticklabels(eval_datasets, fontsize=XTICK_SIZE, rotation=25, ha='right')
        ax.tick_params(axis='y', labelsize=TICK_SIZE)
        ax.grid(True, linestyle='-', alpha=0.6, color='#B0B0B0', linewidth=0.8, axis='y')
        ax.set_axisbelow(True)
    
    # Add centered legend for right section
    right_center = right_start + right_available / 2
    from matplotlib.patches import Patch
    legend_elements_bar = [
        Patch(facecolor=COLORS['Pretrained'], label='Qwen2.5-Math-1.5B'),
        Patch(facecolor=COLORS['TTRL'], label='TTRL'),
        Patch(facecolor=COLORS['TTCS'], label='TTCS (Ours)'),
    ]
    fig.legend(handles=legend_elements_bar, loc='upper center', bbox_to_anchor=(right_center, 0.97), 
               ncol=3, fontsize=LEGEND_SIZE, frameon=False)
    
    # Add subfigure captions - positioned below x-axis labels
    caption_y = 0.11
    caption_y2 = 0.20

    fig.text(left_center, caption_y, '(a) General-domain performance comparison', 
             fontsize=LABEL_SIZE, ha='center', va='top')
    fig.text(left_center, caption_y2, 'Checkpoint Stage', 
             fontsize=LABEL_SIZE, ha='center', va='top')

    fig.text(right_center, caption_y, '(b) Mathematical-domain performance comparison', 
             fontsize=LABEL_SIZE, ha='center', va='top')
    fig.text(right_center, caption_y2, 'Evaluation Dataset', 
             fontsize=LABEL_SIZE, ha='center', va='top')

    # Save
    output_path = os.path.join(OUTPUT_DIR, "combined_figure_2x2.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    output_pdf = os.path.join(OUTPUT_DIR, "combined_figure_2x2.pdf")
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {output_pdf}")
    
    plt.close()


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("\nCollecting line chart data (AIME25)...")
    line_data = collect_line_chart_data()
    
    print("\nCollecting bar chart data...")
    bar_data = collect_bar_chart_data()
    
    print("\nGenerating combined figure (3x3)...")
    plot_combined_figure(line_data, bar_data)
    
    print("\nGenerating combined figure (2x2)...")
    plot_combined_figure_2x2(line_data, bar_data)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
