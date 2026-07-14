"""
Nature 期刊风格数据图绘制脚本。

风格规范：
- 黑白灰基底 + 单色点缀（深蓝 #2166AC / 暗红 #B2182B）
- 去除所有冗余装饰（上/右边框、网格、背景色）
- 坐标轴标题首字母大写（如 "Episode" 而非 "回合"）
- 图例无边框
- 刻度朝外
- 每图独立保存，对应报告特定章节
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# ── 全局样式 ──────────────────────────────────────────────
rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "SimSun"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# ── 颜色 ──────────────────────────────────────────────────
C_BLUE   = "#2166AC"   # 深蓝（主色）
C_RED    = "#B2182B"   # 暗红（次要色）
C_GRAY   = "#666666"   # 灰色（文本/标注）
C_LGRAY  = "#CCCCCC"   # 浅灰
C_BLACK  = "#000000"   # 黑色

RESULTS_DIR = "results_task"

def _save(fig, name):
    """同时保存 PDF 和 PNG"""
    fig.savefig(f"{RESULTS_DIR}/{name}.pdf")
    fig.savefig(f"{RESULTS_DIR}/{name}.png", dpi=200)
    plt.close()
    print(f"[OK] {name}")


def _smooth(data, window=20):
    """移动平均平滑"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def fig1_ppo_training():
    """图1：PPO训练奖励曲线（对应报告2.2节）"""
    with open(f"{RESULTS_DIR}/ppo_train_log.json") as f:
        data = json.load(f)

    rewards = data["episode_rewards"]
    episodes = np.arange(1, len(rewards) + 1)
    smoothed = _smooth(rewards, window=10)

    fig, ax = plt.subplots(figsize=(4.5, 3))

    # 原始数据（浅色散点）
    ax.scatter(episodes, rewards, s=1.5, color=C_BLUE, alpha=0.25, linewidth=0,
               label="Reward per episode")

    # 平滑曲线
    ax.plot(np.arange(10, len(rewards) + 1), smoothed,
            color=C_BLUE, linewidth=1.2, label="Moving average (w=10)")

    # 满分阈值线
    ax.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.text(len(rewards) * 0.72, 483, "Solved (475)", fontsize=6.5,
            color=C_GRAY, fontstyle="italic")

    # 标注首次满分
    first_solved = None
    for i, r in enumerate(rewards):
        if r >= 475:
            first_solved = i + 1
            break
    if first_solved:
        ax.axvline(x=first_solved, color=C_RED, linestyle=":", linewidth=0.7, alpha=0.5)
        ax.text(first_solved + 8, 100, f"Episode {first_solved}",
                fontsize=6.5, color=C_RED, fontstyle="italic")

    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel("Reward", fontsize=8)
    ax.set_title("Figure 1 | PPO training curve", fontsize=9, fontweight="bold", pad=8)

    ax.set_xlim(0, len(rewards) + 5)
    ax.set_ylim(-50, 550)
    ax.legend(loc="lower right")

    _save(fig, "fig1_ppo_training")


def fig2_dqn_training():
    """图2：DQN训练奖励曲线——展示灾难性遗忘（对应报告2.3节）"""
    with open(f"{RESULTS_DIR}/dqn_train_log.json") as f:
        data = json.load(f)

    rewards = data["episode_rewards"]
    episodes = np.arange(1, len(rewards) + 1)
    smoothed = _smooth(rewards, window=50)

    fig, ax = plt.subplots(figsize=(5.5, 3))

    # 原始数据
    ax.scatter(episodes, rewards, s=0.5, color=C_GRAY, alpha=0.15, linewidth=0)

    # 平滑曲线
    ax.plot(np.arange(50, len(rewards) + 1), smoothed,
            color=C_RED, linewidth=1.0, label="Moving average (w=50)")

    # 标注三个阶段
    phases = [
        (0, 1200, "Exploration", "#888888"),
        (1200, 2200, "Learning", C_BLUE),
        (2200, len(rewards), "Catastrophic\nforgetting", C_RED),
    ]
    for start, end, label, color in phases:
        mid = (start + end) / 2
        y_pos = 420 if label == "Learning" else 380
        ax.annotate(label, xy=(mid, y_pos), fontsize=6.5, color=color,
                    ha="center", fontstyle="italic",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.5))

    # 峰值标注
    peak_idx = int(np.argmax(rewards))
    peak_val = max(rewards)
    ax.scatter([peak_idx + 1], [peak_val], s=30, color=C_RED, zorder=5,
               edgecolors="white", linewidth=0.5)
    ax.annotate(f"Peak: {peak_val:.0f}", xy=(peak_idx + 1, peak_val),
                xytext=(peak_idx - 800, peak_val + 60),
                fontsize=6.5, color=C_RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.6))

    # 满分线
    ax.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(100, 482, "Solved (475)", fontsize=6, color=C_GRAY, fontstyle="italic")

    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel("Reward", fontsize=8)
    ax.set_title("Figure 2 | DQN training curve — catastrophic forgetting",
                 fontsize=9, fontweight="bold", pad=8)

    ax.set_xlim(-50, len(rewards) + 50)
    ax.set_ylim(-50, 550)
    ax.legend(loc="upper left")

    _save(fig, "fig2_dqn_training")


def fig3_method_comparison():
    """图3：五种控制方法性能对比（对应报告3.3节）"""
    with open(f"{RESULTS_DIR}/baseline_comparison.json") as f:
        baseline = json.load(f)
    with open(f"{RESULTS_DIR}/rl_eval.json") as f:
        rl = json.load(f)

    # 数据准备
    labels = ["Random", "LQR", "Rule-\nbased", "DQN", "PPO"]
    means = [
        baseline["random"]["mean_reward"],
        baseline["lqr"]["mean_reward"],
        baseline["rule_based"]["mean_reward"],
        rl["dqn"]["mean_reward"],
        rl["ppo"]["mean_reward"],
    ]
    errors = [0, 0, 0, rl["dqn"]["std_reward"], rl["ppo"]["std_reward"]]
    colors = [C_GRAY, "#999999", "#555555", C_RED, C_BLUE]
    hatch = ["", "", "", "//", ""]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    x = np.arange(len(labels))
    bars = ax.bar(x, means, width=0.55, color=colors, edgecolor=C_BLACK,
                  linewidth=0.5, alpha=0.85)

    # 误差棒
    for i, (bar, err) in enumerate(zip(bars, errors)):
        if err > 0:
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        yerr=err, fmt="none", color=C_BLACK, capsize=2.5,
                        capthick=0.6, linewidth=0.6)

    # 数值标注
    for bar, mean in zip(bars, means):
        y_pos = bar.get_height() + 15
        if mean < 10:
            y_pos = bar.get_height() + 8
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=7, color=C_BLACK)

    # 满分线
    ax.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(3.8, 482, "Solved (475)", fontsize=6.5, color=C_GRAY, fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean reward", fontsize=8)
    ax.set_title("Figure 3 | Control method comparison",
                 fontsize=9, fontweight="bold", pad=8)
    ax.set_ylim(0, 580)

    _save(fig, "fig3_method_comparison")


def fig4_training_comparison():
    """图4：PPO vs DQN 训练全过程对比（对应报告3.1节）"""
    with open(f"{RESULTS_DIR}/ppo_train_log.json") as f:
        ppo = json.load(f)
    with open(f"{RESULTS_DIR}/dqn_train_log.json") as f:
        dqn = json.load(f)

    ppo_rewards = ppo["episode_rewards"]
    dqn_rewards = dqn["episode_rewards"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)

    # ── PPO ──
    ppo_sm = _smooth(ppo_rewards, 10)
    ax1.scatter(range(1, len(ppo_rewards) + 1), ppo_rewards,
                s=1, color=C_BLUE, alpha=0.2, linewidth=0)
    ax1.plot(range(10, len(ppo_rewards) + 1), ppo_sm,
             color=C_BLUE, linewidth=1.2)
    ax1.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.6, alpha=0.5)
    ax1.set_xlabel("Episode", fontsize=7.5)
    ax1.set_ylabel("Reward", fontsize=7.5)
    ax1.set_title("PPO", fontsize=8, fontweight="bold", pad=5)
    ax1.set_ylim(-50, 550)

    # ── DQN ──
    dqn_sm = _smooth(dqn_rewards, 50)
    ax2.scatter(range(1, len(dqn_rewards) + 1), dqn_rewards,
                s=0.3, color=C_RED, alpha=0.08, linewidth=0)
    ax2.plot(range(50, len(dqn_rewards) + 1), dqn_sm,
             color=C_RED, linewidth=1.0)
    ax2.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.6, alpha=0.5)
    ax2.set_xlabel("Episode", fontsize=7.5)
    ax2.set_title("DQN (catastrophic forgetting)", fontsize=8,
                  fontweight="bold", pad=5, color=C_RED)
    ax2.set_ylim(-50, 550)

    fig.suptitle("Figure 4 | Training process comparison",
                 fontsize=9, fontweight="bold", y=1.04)

    _save(fig, "fig4_training_comparison")


def fig5_before_after():
    """图5：训练前后控制效果对比（箱线图，对应报告3.2节）"""
    with open(f"{RESULTS_DIR}/rl_eval.json") as f:
        rl = json.load(f)

    # 随机策略数据（从baseline读）
    with open(f"{RESULTS_DIR}/baseline_comparison.json") as f:
        bl = json.load(f)

    random_rewards = bl["random"]["rewards"][:20]
    dqn_rewards = rl["dqn"]["rewards"]
    ppo_rewards = rl["ppo"]["rewards"]

    fig, ax = plt.subplots(figsize=(3.5, 3))

    data = [random_rewards, dqn_rewards, ppo_rewards]
    labels = ["Random\n(before)", "DQN\n(after)", "PPO\n(after)"]
    colors = [C_GRAY, C_RED, C_BLUE]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.45,
                    medianprops=dict(color="white", linewidth=1.2),
                    whiskerprops=dict(color=C_BLACK, linewidth=0.6),
                    capprops=dict(color=C_BLACK, linewidth=0.6),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor=C_GRAY, alpha=0.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(2.8, 482, "Solved (475)", fontsize=6.5, color=C_GRAY, fontstyle="italic")

    ax.set_ylabel("Reward", fontsize=8)
    ax.set_title("Figure 5 | Before vs after training",
                 fontsize=9, fontweight="bold", pad=8)

    _save(fig, "fig5_before_after")


if __name__ == "__main__":
    print("Generating Nature-style figures...\n")
    fig1_ppo_training()
    fig2_dqn_training()
    fig3_method_comparison()
    fig4_training_comparison()
    fig5_before_after()
    print("\nAll figures generated in", RESULTS_DIR)
