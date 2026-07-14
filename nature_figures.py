"""
Nature 期刊风格数据图绘制脚本（A2C 替代 DQN 版本）。

每图独立保存，对应报告特定章节。
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

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

C_BLUE  = "#2166AC"
C_RED   = "#B2182B"
C_GREEN = "#1B7837"
C_GRAY  = "#666666"
C_BLACK = "#000000"
RESULTS_DIR = "results_task"

def _save(fig, name):
    fig.savefig(f"{RESULTS_DIR}/{name}.pdf")
    fig.savefig(f"{RESULTS_DIR}/{name}.png", dpi=200)
    plt.close()
    print(f"[OK] {name}")

def _smooth(data, window=20):
    if len(data) < window: return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


# ═══════════════════════════════════════════════════════════
# Figure 1: PPO training curve (Section 2.2)
# ═══════════════════════════════════════════════════════════
def fig1_ppo_training():
    with open(f"{RESULTS_DIR}/ppo_train_log.json") as f:
        data = json.load(f)
    rewards = data["episode_rewards"]
    lengths = data.get("episode_lengths", [])
    episodes = np.arange(1, len(rewards) + 1)
    smoothed = _smooth(rewards, 10)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.scatter(episodes, rewards, s=1.5, color=C_BLUE, alpha=0.2, linewidth=0,
               label="Reward per episode")
    ax.plot(np.arange(10, len(rewards)+1), smoothed,
            color=C_BLUE, linewidth=1.2, label="Moving average (w=10)")
    ax.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(len(rewards)*0.72, 483, "Solved (475)", fontsize=6.5,
            color=C_GRAY, fontstyle="italic")

    first_solved = next((i+1 for i,r in enumerate(rewards) if r>=475), None)
    if first_solved:
        ax.axvline(x=first_solved, color=C_RED, linestyle=":", linewidth=0.7, alpha=0.5)
        ax.text(first_solved+8, 100, f"Episode {first_solved}",
                fontsize=6.5, color=C_RED, fontstyle="italic")

    # Statistical annotations
    last100 = np.mean(rewards[-100:])
    stats = f"Last 100 mean: {last100:.0f}\nMax: {max(rewards):.0f}\nEpisodes: {len(rewards)}"
    ax.text(0.97, 0.35, stats, transform=ax.transAxes, fontsize=6.5,
            color=C_GRAY, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_GRAY, linewidth=0.3))

    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel("Reward", fontsize=8)
    ax.set_title("PPO training curve", fontsize=9, fontweight="bold", pad=8)
    ax.set_xlim(0, len(rewards)+5); ax.set_ylim(-50, 550)
    ax.legend(loc="lower right")
    _save(fig, "fig1_ppo_training")


# ═══════════════════════════════════════════════════════════
# Figure 2: A2C training curve (Section 2.3)
# ═══════════════════════════════════════════════════════════
def fig2_a2c_training():
    with open(f"{RESULTS_DIR}/a2c_train_log.json") as f:
        data = json.load(f)
    rewards = data["episode_rewards"]
    episodes = np.arange(1, len(rewards)+1)
    smoothed = _smooth(rewards, 50)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.scatter(episodes, rewards, s=0.5, color=C_GREEN, alpha=0.15, linewidth=0)
    ax.plot(np.arange(50, len(rewards)+1), smoothed,
            color=C_GREEN, linewidth=1.0, label="Moving average (w=50)")

    # Annotate: first time reaching 500
    first_500 = next((i+1 for i,r in enumerate(rewards) if r>=475), None)
    if first_500:
        ax.axvline(x=first_500, color=C_GRAY, linestyle=":", linewidth=0.7, alpha=0.5)
        ax.text(first_500+20, 450, f"First solved: Episode {first_500}",
                fontsize=6.5, color=C_GRAY, fontstyle="italic")

    # Annotate peak area
    peak_idx = np.argmax(rewards)
    if rewards[peak_idx] >= 475:
        ax.scatter([peak_idx+1], [rewards[peak_idx]], s=30, color=C_GREEN, zorder=5,
                   edgecolors="white", linewidth=0.5)
        ax.annotate(f"Peak: {rewards[peak_idx]:.0f}", xy=(peak_idx+1, rewards[peak_idx]),
                    xytext=(peak_idx-300, rewards[peak_idx]+60),
                    fontsize=6.5, color=C_GREEN, ha="center",
                    arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=0.6))

    ax.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(100, 482, "Solved (475)", fontsize=6, color=C_GRAY, fontstyle="italic")

    # Statistics
    last100 = np.mean(rewards[-100:])
    solved_count = sum(1 for r in rewards if r>=475)
    stats = f"Last 100 mean: {last100:.0f}\nSolved episodes: {solved_count}/{len(rewards)}\nMax: {max(rewards):.0f}"
    ax.text(0.97, 0.45, stats, transform=ax.transAxes, fontsize=6.5,
            color=C_GRAY, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_GRAY, linewidth=0.3))

    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel("Reward", fontsize=8)
    ax.set_title("A2C training curve", fontsize=9, fontweight="bold", pad=8)
    ax.set_xlim(-50, len(rewards)+50); ax.set_ylim(-50, 550)
    ax.legend(loc="upper left")
    _save(fig, "fig2_a2c_training")


# ═══════════════════════════════════════════════════════════
# Figure 3: Method comparison bar chart (Section 3.3)
# ═══════════════════════════════════════════════════════════
def fig3_method_comparison():
    with open(f"{RESULTS_DIR}/baseline_comparison.json") as f:
        bl = json.load(f)
    with open(f"{RESULTS_DIR}/rl_eval.json") as f:
        rl = json.load(f)

    labels = ["Random", "LQR", "Rule-\nbased", "A2C", "PPO"]
    means = [
        bl["random"]["mean_reward"],
        bl["lqr"]["mean_reward"],
        bl["rule_based"]["mean_reward"],
        rl["a2c"]["mean_reward"],
        rl["ppo"]["mean_reward"],
    ]
    errors = [0, 0, 0, rl["a2c"]["std_reward"], rl["ppo"]["std_reward"]]
    colors = [C_GRAY, "#999999", "#555555", C_GREEN, C_BLUE]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, width=0.55, color=colors, edgecolor=C_BLACK,
                  linewidth=0.5, alpha=0.85)

    for i, (bar, err) in enumerate(zip(bars, errors)):
        if err > 0:
            ax.errorbar(bar.get_x()+bar.get_width()/2, bar.get_height(),
                        yerr=err, fmt="none", color=C_BLACK, capsize=2.5,
                        capthick=0.6, linewidth=0.6)

    for bar, mean in zip(bars, means):
        y_pos = bar.get_height() + (8 if mean<10 else 15)
        ax.text(bar.get_x()+bar.get_width()/2, y_pos,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=7, color=C_BLACK)

    ax.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(3.8, 482, "Solved (475)", fontsize=6.5, color=C_GRAY, fontstyle="italic")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean reward", fontsize=8)
    ax.set_title("Control method comparison",
                 fontsize=9, fontweight="bold", pad=8)
    ax.set_ylim(0, 580)
    _save(fig, "fig3_method_comparison")


# ═══════════════════════════════════════════════════════════
# Figure 4: PPO vs A2C training comparison (Section 3.1)
# ═══════════════════════════════════════════════════════════
def fig4_training_comparison():
    with open(f"{RESULTS_DIR}/ppo_train_log.json") as f:
        ppo = json.load(f)
    with open(f"{RESULTS_DIR}/a2c_train_log.json") as f:
        a2c = json.load(f)

    ppo_r = ppo["episode_rewards"]
    a2c_r = a2c["episode_rewards"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2), sharey=True)

    # PPO
    ppo_sm = _smooth(ppo_r, 10)
    ax1.scatter(range(1, len(ppo_r)+1), ppo_r, s=1, color=C_BLUE, alpha=0.2, linewidth=0)
    ax1.plot(range(10, len(ppo_r)+1), ppo_sm, color=C_BLUE, linewidth=1.2)
    ax1.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.6, alpha=0.5)
    solved_ppo = sum(1 for r in ppo_r if r>=475)
    ax1.text(0.97, 0.95, f"Solved: {solved_ppo}/{len(ppo_r)}\nLast100: {np.mean(ppo_r[-100:]):.0f}",
             transform=ax1.transAxes, fontsize=6.5, color=C_BLUE, ha="right", va="top")
    ax1.set_xlabel("Episode", fontsize=7.5); ax1.set_ylabel("Reward", fontsize=7.5)
    ax1.set_title("PPO", fontsize=8, fontweight="bold", pad=5); ax1.set_ylim(-50, 550)

    # A2C
    a2c_sm = _smooth(a2c_r, 50)
    ax2.scatter(range(1, len(a2c_r)+1), a2c_r, s=0.3, color=C_GREEN, alpha=0.1, linewidth=0)
    ax2.plot(range(50, len(a2c_r)+1), a2c_sm, color=C_GREEN, linewidth=1.0)
    ax2.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.6, alpha=0.5)
    solved_a2c = sum(1 for r in a2c_r if r>=475)
    ax2.text(0.97, 0.95, f"Solved: {solved_a2c}/{len(a2c_r)}\nLast100: {np.mean(a2c_r[-100:]):.0f}",
             transform=ax2.transAxes, fontsize=6.5, color=C_GREEN, ha="right", va="top")
    ax2.set_xlabel("Episode", fontsize=7.5)
    ax2.set_title("A2C", fontsize=8, fontweight="bold", pad=5); ax2.set_ylim(-50, 550)

    fig.suptitle("Training process comparison",
                 fontsize=9, fontweight="bold", y=1.04)
    _save(fig, "fig4_training_comparison")


# ═══════════════════════════════════════════════════════════
# Figure 5: Before vs after (Section 3.2)
# ═══════════════════════════════════════════════════════════
def fig5_before_after():
    with open(f"{RESULTS_DIR}/rl_eval.json") as f:
        rl = json.load(f)
    with open(f"{RESULTS_DIR}/baseline_comparison.json") as f:
        bl = json.load(f)

    data = [bl["random"]["rewards"][:20], rl["a2c"]["rewards"], rl["ppo"]["rewards"]]
    labels = ["Random\n(before)", "A2C\n(after)", "PPO\n(after)"]
    colors = [C_GRAY, C_GREEN, C_BLUE]

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.45,
                    medianprops=dict(color="white", linewidth=1.2),
                    whiskerprops=dict(color=C_BLACK, linewidth=0.6),
                    capprops=dict(color=C_BLACK, linewidth=0.6),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=C_GRAY, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    # Add scatter points for individual episodes
    for i, (d, c) in enumerate(zip(data, colors)):
        jitter = np.random.normal(0, 0.04, len(d))
        ax.scatter(np.ones(len(d))*(i+1)+jitter, d, s=8, color=c, alpha=0.4, zorder=3)

    ax.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(2.8, 482, "Solved (475)", fontsize=6.5, color=C_GRAY, fontstyle="italic")
    ax.set_ylabel("Reward", fontsize=8)
    ax.set_title("Before vs after training",
                 fontsize=9, fontweight="bold", pad=8)
    _save(fig, "fig5_before_after")


# ═══════════════════════════════════════════════════════════
# Figure 6: Learning metrics panel (Section 3.1 multi-metric)
# ═══════════════════════════════════════════════════════════
def fig6_metrics_panel():
    """多指标对比面板：奖励曲线、平均奖励、成功率、回合步数"""
    with open(f"{RESULTS_DIR}/ppo_train_log.json") as f:
        ppo = json.load(f)
    with open(f"{RESULTS_DIR}/a2c_train_log.json") as f:
        a2c = json.load(f)

    ppo_r, ppo_l = ppo["episode_rewards"], ppo.get("episode_lengths", [])
    a2c_r, a2c_l = a2c["episode_rewards"], a2c.get("episode_lengths", [])

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

    # (a) Reward curves
    ax = axes[0, 0]
    wins = [50, 50]
    for name, data, color, w in zip(["PPO", "A2C"], [ppo_r, a2c_r], [C_BLUE, C_GREEN], [10, 50]):
        sm = _smooth(data, w)
        ax.plot(range(w, len(data)+1), sm, color=color, linewidth=1.0, label=name)
    ax.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.set_title("(a) Training reward curves", fontsize=8, fontweight="bold")
    ax.legend(loc="lower right"); ax.set_ylim(-50, 550)

    # (b) Rolling average reward (100-episode window)
    ax = axes[0, 1]
    for name, data, color in zip(["PPO", "A2C"], [ppo_r, a2c_r], [C_BLUE, C_GREEN]):
        roll = np.convolve(data, np.ones(100)/100, mode="valid")
        ax.plot(range(100, len(data)+1), roll, color=color, linewidth=1.0, label=name)
    ax.axhline(y=475, color=C_GRAY, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Episode"); ax.set_ylabel("Rolling mean (w=100)")
    ax.set_title("(b) Rolling average reward", fontsize=8, fontweight="bold")
    ax.legend(loc="lower right")

    # (c) Success rate (rolling 100-episode window, threshold=475)
    ax = axes[1, 0]
    for name, data, color in zip(["PPO", "A2C"], [ppo_r, a2c_r], [C_BLUE, C_GREEN]):
        success = np.array([1 if r>=475 else 0 for r in data], dtype=float)
        rate = np.convolve(success, np.ones(100)/100, mode="valid")
        ax.plot(range(100, len(data)+1), rate*100, color=color, linewidth=1.0, label=name)
    ax.set_xlabel("Episode"); ax.set_ylabel("Success rate (%)")
    ax.set_title("(c) Rolling success rate (w=100)", fontsize=8, fontweight="bold")
    ax.legend(loc="lower right"); ax.set_ylim(0, 105)

    # (d) Episode length
    ax = axes[1, 1]
    for name, lengths, color in zip(["PPO", "A2C"], [ppo_l, a2c_l], [C_BLUE, C_GREEN]):
        if lengths:
            sm = _smooth(lengths, 20)
            ax.plot(range(20, len(lengths)+1), sm, color=color, linewidth=1.0, label=name)
    ax.set_xlabel("Episode"); ax.set_ylabel("Episode length (steps)")
    ax.set_title("(d) Episode length", fontsize=8, fontweight="bold")
    ax.legend(loc="lower right")

    fig.suptitle("Multi-metric training analysis",
                 fontsize=9, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "fig6_metrics_panel")


if __name__ == "__main__":
    print("Generating Nature-style figures (A2C version)...\n")
    fig1_ppo_training()
    fig2_a2c_training()
    fig3_method_comparison()
    fig4_training_comparison()
    fig5_before_after()
    fig6_metrics_panel()
    print("\nAll figures generated in", RESULTS_DIR)
