import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path

def draw_professional_rise_architecture():
    # 1. 设置画布与风格
    fig, ax = plt.figure(figsize=(22, 11), dpi=150), plt.gca()
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # 定义专业学术配色
    c_bg_stage1 = '#e8f0fe' # 浅蓝背景
    c_bg_stage2 = '#e6f4ea' # 浅绿背景
    c_bg_stage3 = '#f3e5f5' # 浅紫背景
    
    c_node_data = '#ffffff'    # 数据节点 (白底深框)
    c_node_agent_syn = '#b3e5fc' # 合成器 Agent (亮蓝)
    c_node_agent_sol = '#c8e6c9' # 解题器 Agent (亮绿)
    c_node_process = '#fff9c4'   # 过程/过滤器 (浅黄)
    c_node_reward = '#ffccbc'    # 奖励 (浅橙)
    
    c_border_light = '#cfd8dc'
    c_border_dark = '#455a64'
    c_text_title = '#263238'
    c_text_body = '#37474f'
    c_arrow = '#546e7a'
    
    font_family = 'DejaVu Sans' # 确保跨平台兼容性

    # ================== 辅助绘制函数 ==================
    def draw_fancy_box(x, y, w, h, text, color, border_color=c_border_dark, subtext=None, style='round', label_top=None):
        if style == 'round':
            box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.3", 
                                 linewidth=1.5, edgecolor=border_color, facecolor=color, zorder=2)
            ax.add_patch(box)
        elif shape == 'diamond': # Diamond for decision/filter
            pts = [(x+w/2, y), (x+w, y+h/2), (x+w/2, y+h), (x, y+h/2)]
            box = plt.Polygon(pts, closed=True, linewidth=1.5, edgecolor=border_color, facecolor=color, zorder=2)
            ax.add_patch(box)
        elif shape == 'circle': # Circle for reward
            box = plt.Circle((x+w/2, y+h/2), radius=min(w,h)/2, linewidth=1.5, edgecolor=border_color, facecolor=color, zorder=2)
            ax.add_patch(box)
            
        cx, cy = x + w/2, y + h/2
        if label_top:
            ax.text(cx, y+h+0.25, label_top, ha='center', va='bottom', fontsize=10, fontweight='bold', color=c_text_title, fontfamily=font_family)
        ax.text(cx, cy + (0.15 if subtext else 0), text, ha='center', va='center', fontsize=12, fontweight='bold', color=c_text_body, fontfamily=font_family, zorder=3)
        if subtext:
            ax.text(cx, cy - 0.3, subtext, ha='center', va='center', fontsize=9, color=c_text_body, fontfamily=font_family, style='italic', zorder=3)
        return box

    def draw_arrow_curve(x1, y1, x2, y2, label=None, rad=0.0, width=2, color=c_arrow, ls='-'):
        style = f"Simple, tail_width={width}, head_width={width*3}, head_length={width*2.5}"
        arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}", 
                                        arrowstyle=style, color=color, lw=0, linestyle=ls, zorder=1)
        ax.add_patch(arrow)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            # Simple offset for labels based on arc direction
            off_y = 0.3 if rad >= 0 else -0.5
            ax.text(mx, my+off_y, label, ha='center', va='center', fontsize=10, fontweight='bold', color=color, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # ================== 1. 绘制背景区域 (Stages) ==================
    # Stage 1: Synthesis
    bg1 = FancyBboxPatch((0.5, 0.5), 6.5, 10, boxstyle="round,pad=0.2,rounding_size=0.5", linewidth=2, edgecolor='#bbdefb', facecolor=c_bg_stage1, alpha=0.8, zorder=0)
    ax.add_patch(bg1)
    ax.text(3.75, 10.2, "STAGE 1: Structural Synthesis", ha='center', fontsize=14, fontweight='bold', color='#1565c0', fontfamily=font_family)

    # Stage 2: Evaluation
    bg2 = FancyBboxPatch((7.5, 0.5), 6, 10, boxstyle="round,pad=0.2,rounding_size=0.5", linewidth=2, edgecolor='#c8e6c9', facecolor=c_bg_stage2, alpha=0.8, zorder=0)
    ax.add_patch(bg2)
    ax.text(10.5, 10.2, "STAGE 2: Evaluation & Reward", ha='center', fontsize=14, fontweight='bold', color='#2e7d32', fontfamily=font_family)

    # Stage 3: Optimization
    bg3 = FancyBboxPatch((14, 0.5), 7.5, 10, boxstyle="round,pad=0.2,rounding_size=0.5", linewidth=2, edgecolor='#e1bee7', facecolor=c_bg_stage3, alpha=0.8, zorder=0)
    ax.add_patch(bg3)
    ax.text(17.75, 10.2, "STAGE 3: Transductive Optimization", ha='center', fontsize=14, fontweight='bold', color='#6a1b9a', fontfamily=font_family)

    # ================== 2. 绘制节点 (Nodes) ==================

    # --- Stage 1 Nodes ---
    node_seed = draw_fancy_box(1.2, 8, 2, 1.2, "Test Seed\n$x_{ref}$", c_node_data, label_top="Input")
    node_syn = draw_fancy_box(4.2, 8, 2.2, 1.4, "Synthesizer\nAgent ($\pi_\phi$)", c_node_agent_syn, subtext="ICL w/ Constraints")
    
    # Constraint Details (Small box attached to Synthesizer)
    draw_fancy_box(4.3, 6.5, 2, 1.1, "Constraints:\nIsomorphism\nObject Mapping", '#e3f2fd', border_color=c_node_agent_syn, style='round')
    draw_arrow_curve(5.3, 7.6, 5.3, 8, width=1, color=c_node_agent_syn, ls=':')

    node_raw = draw_fancy_box(4.2, 4.5, 2.2, 1.2, "Raw Queries\n$\{x'_k\}_{raw}$", c_node_data)
    
    # Sim Filter (Diamond Shape)
    shape = 'diamond'
    node_filter = draw_fancy_box(4.8, 2, 1.6, 1.6, "Sim\nFilter", c_node_process, subtext="Skel & Jaccard")

    # --- Stage 2 Nodes ---
    node_filtered_data = draw_fancy_box(8.5, 4.5, 2.2, 1.2, "Filtered\nDataset $\mathcal{D}_{syn}$", c_node_data)
    
    # Solver Eval (Agent)
    node_solver_eval = draw_fancy_box(11.2, 4.5, 2, 1.4, "Solver Eval\n($\pi_\theta$)", c_node_agent_sol, subtext="Rollout & Vote")

    # Bell Reward (Circle Shape)
    shape = 'circle'
    node_reward = draw_fancy_box(11.4, 7.5, 1.6, 1.6, "Bell Reward\n$R(s)$", c_node_reward, subtext="Variance-Driven")
    shape = 'round' # reset

    # --- Stage 3 Nodes ---
    node_real_data = draw_fancy_box(14.5, 8, 2, 1.2, "Real Data\n$\mathcal{D}_{real}$", c_node_data, label_top="Prior Knowledge")
    
    # Merge Process
    node_merge = draw_fancy_box(15.8, 5.8, 2, 1.2, "Hybrid\nMerge", c_node_process, subtext="Front-Loading")
    
    # DAPO Filter (Diamond)
    shape = 'diamond'
    node_dapo = draw_fancy_box(15.8, 3.2, 1.8, 1.8, "DAPO\nFilter", c_node_process, subtext="Difficulty Check")
    shape = 'round'

    node_train_batch = draw_fancy_box(18.5, 3.5, 2, 1.2, "Train Batch\n$\mathcal{D}_{train}$", c_node_data)

    # GRPO Optimization Process
    node_grpo = draw_fancy_box(18.5, 1, 2.2, 1.4, "GRPO\nOptimization", '#ffecb3', border_color='#ffb300', subtext="Maximize Advantage")
    
    # Solver Update (Agent)
    node_solver_update = draw_fancy_box(15, 1, 2.5, 1.4, "Solver Update\n($\\theta \\to \\theta^*$)", c_node_agent_sol, border_color='#2e7d32')

    # ================== 3. 绘制连接流 (Data Flow Arrows) ==================

    # S1 Flow
    draw_arrow_curve(3.2, 8.6, 4.2, 8.6, label="Guide")
    draw_arrow_curve(5.3, 8, 5.3, 5.7, label="Generate")
    draw_arrow_curve(5.3, 4.5, 5.6, 3.6, width=1.5) # Raw -> Filter
    
    # Filter Logic
    draw_arrow_curve(4.8, 2.8, 4.0, 2.8, label="Reject", color='#d32f2f', rad=-0.2, ls='--') # Reject Path
    draw_arrow_curve(6.4, 2.8, 8.5, 4.8, label="Pass Check", color='#2e7d32', rad=-0.3, width=2.5) # Pass Path (Cross-Stage)

    # S2 Flow
    draw_arrow_curve(10.7, 5.1, 11.2, 5.1, width=1.5) # Data -> Eval
    draw_arrow_curve(12.2, 5.9, 12.2, 7.5, label="Pass Rate $s$") # Eval -> Reward

    # Reward Feedback (Important Loop)
    draw_arrow_curve(11.4, 8.3, 6.4, 8.3, label="Guidance Signal (Implicit)", color='#ef6c00', rad=0.15, ls='--', width=1.5)

    # S3 Flow
    draw_arrow_curve(10.7, 4.8, 15.8, 6.1, label="Synthetic", rad=-0.1) # Syn -> Merge
    draw_arrow_curve(15.5, 8, 16.3, 7, label="Sampled", rad=0.1) # Real -> Merge
    draw_arrow_curve(16.8, 5.8, 16.8, 5, width=1.5) # Merge -> DAPO
    
    # DAPO Logic
    draw_arrow_curve(17.6, 4.1, 18.5, 4.1, label="Keep", color='#2e7d32') # DAPO -> Train
    draw_arrow_curve(15.8, 4.1, 14.8, 4.1, label="Discard", color='#d32f2f', ls='--') # DAPO Reject

    # Optimization Loop (The core cycle)
    draw_arrow_curve(19.5, 3.5, 19.5, 2.4, width=1.5) # Batch -> GRPO
    draw_arrow_curve(18.5, 1.7, 17.5, 1.7, label="Update $\\theta^*$", color='#d32f2f', width=2.5) # GRPO -> Solver Update
    
    # Solver closing the loop (conceptually, the updated solver is used in next steps)
    draw_arrow_curve(15, 1.7, 13, 4.5, label="Next Iteration", color=c_node_agent_sol, rad=0.4, ls='--', width=1.5)

    # Final Touches
    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig

# 生成图像
fig = draw_professional_rise_architecture()
plt.show()
plt.savefig('professional_rise_architecture.png')