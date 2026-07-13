#!/usr/bin/env python
"""生成所有可视化图表。"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _smooth(values, window=10):
    """移动平均平滑。"""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curves():
    """绘制 DQN vs PPO 训练奖励曲线。"""
    print("\n绘制训练奖励曲线...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, algo in enumerate(["dqn", "ppo"]):
        log_path = os.path.join(RESULTS_DIR, f"{algo}_train_log.json")
        if not os.path.exists(log_path):
            print(f"  ⚠️  {algo} 训练日志不存在: {log_path}")
            continue

        with open(log_path) as f:
            data = json.load(f)

        rewards = data["episode_rewards"]

        ax = axes[idx]
        # 原始奖励（半透明）
        ax.plot(rewards, alpha=0.3, color="steelblue", linewidth=0.5, label="回合奖励")
        # 平滑曲线
        if len(rewards) >= 20:
            smooth_rewards = _smooth(rewards, window=20)
            ax.plot(range(19, len(rewards)), smooth_rewards,
                    color="darkblue", linewidth=2, label="移动平均 (w=20)")
        # 满分线
        ax.axhline(y=500, color="green", linestyle="--", alpha=0.5, label="满分 (500)")
        # 解决线
        ax.axhline(y=475, color="orange", linestyle="--", alpha=0.5, label="解决阈值 (475)")

        ax.set_xlabel("回合 (Episode)")
        ax.set_ylabel("奖励 (Reward)")
        ax.set_title(f"{algo.upper()} 训练奖励曲线")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def plot_param_sweep():
    """绘制参数搜索对比图。"""
    print("\n绘制参数搜索对比图...")

    for algo in ["dqn", "ppo"]:
        sweep_path = os.path.join(RESULTS_DIR, f"{algo}_sweep_results.json")
        if not os.path.exists(sweep_path):
            print(f"  ⚠️  {algo} 参数搜索结果不存在")
            continue

        with open(sweep_path) as f:
            data = json.load(f)

        # 找到学习率搜索数据
        lr_key = "learning_rate_sweep"
        if lr_key not in data:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["steelblue", "darkred", "forestgreen"]

        for i, (label, rewards) in enumerate(data[lr_key].items()):
            if len(rewards) >= 10:
                smooth = _smooth(rewards, window=10)
                ax.plot(range(9, len(rewards)), smooth,
                        color=colors[i % len(colors)], linewidth=2, label=label)

        ax.set_xlabel("回合 (Episode)")
        ax.set_ylabel("奖励 (Reward)")
        ax.set_title(f"{algo.upper()} 学习率参数搜索对比")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f"{algo}_lr_sweep.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ 已保存: {save_path}")


def plot_baseline_comparison():
    """绘制传统控制 vs RL 对比柱状图。"""
    print("\n绘制基线对比图...")

    comp_path = os.path.join(RESULTS_DIR, "baseline_comparison.json")
    if not os.path.exists(comp_path):
        print(f"  ⚠️  基线对比结果不存在")
        return

    with open(comp_path) as f:
        data = json.load(f)

    labels = []
    means = []
    stds = []
    colors_list = []

    color_map = {
        "random": "gray",
        "rule_based": "orange",
        "lqr": "purple",
        "dqn": "steelblue",
        "ppo": "green",
    }

    for key in ["random", "rule_based", "lqr", "dqn", "ppo"]:
        if key in data:
            labels.append(data[key]["label"])
            means.append(data[key]["mean_reward"])
            stds.append(data[key].get("std_reward", 0))
            colors_list.append(color_map.get(key, "gray"))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, means, yerr=stds, color=colors_list, edgecolor="black",
                  alpha=0.8, capsize=5)

    # 在柱子上标注数值
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{mean:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.axhline(y=475, color="red", linestyle="--", alpha=0.5, label="解决阈值 (475)")
    ax.set_ylabel("平均奖励 (Mean Reward)")
    ax.set_title("控制方法对比：随机策略 vs 规则 vs LQR vs DQN vs PPO")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0, top=max(means) * 1.15)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "baseline_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def plot_before_after():
    """绘制训练前（随机策略）vs 训练后（智能体）控制效果对比。"""
    print("\n绘制训练前后对比图...")

    import gymnasium as gym

    # 训练前：随机策略
    env = gym.make("CartPole-v1")
    pre_rewards = []
    for ep in range(20):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        pre_rewards.append(total_reward)
    env.close()

    # 训练后：PPO智能体
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "ppo_cartpole")
    post_rewards_ppo = []
    if os.path.exists(model_path + ".zip") or os.path.exists(model_path):
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        env = gym.make("CartPole-v1")
        for ep in range(20):
            obs, _ = env.reset(seed=ep)
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
                total_reward += reward
            post_rewards_ppo.append(total_reward)
        env.close()

    # 训练后：DQN智能体
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "dqn_cartpole")
    post_rewards_dqn = []
    if os.path.exists(model_path + ".zip") or os.path.exists(model_path):
        from stable_baselines3 import DQN
        model = DQN.load(model_path)
        env = gym.make("CartPole-v1")
        for ep in range(20):
            obs, _ = env.reset(seed=ep)
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
                total_reward += reward
            post_rewards_dqn.append(total_reward)
        env.close()

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：柱状图对比
    ax = axes[0]
    categories = ["训练前\n(随机策略)"]
    values = [np.mean(pre_rewards)]
    colors = ["gray"]

    if post_rewards_dqn:
        categories.append("训练后\n(DQN)")
        values.append(np.mean(post_rewards_dqn))
        colors.append("steelblue")
    if post_rewards_ppo:
        categories.append("训练后\n(PPO)")
        values.append(np.mean(post_rewards_ppo))
        colors.append("green")

    bars = ax.bar(categories, values, color=colors, edgecolor="black", alpha=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("平均奖励")
    ax.set_title("训练前后控制效果对比")
    ax.grid(axis="y", alpha=0.3)

    # 右图：回合奖励分布（箱线图）
    ax = axes[1]
    box_data = [pre_rewards]
    box_labels = ["训练前"]
    if post_rewards_dqn:
        box_data.append(post_rewards_dqn)
        box_labels.append("DQN")
    if post_rewards_ppo:
        box_data.append(post_rewards_ppo)
        box_labels.append("PPO")

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("回合奖励")
    ax.set_title("训练前后奖励分布")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "before_after_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def plot_all():
    """生成所有图表。"""
    print("\n" + "=" * 60)
    print("  生成可视化图表")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plot_training_curves()
    plot_param_sweep()
    plot_baseline_comparison()
    plot_before_after()

    print("\n✅ 所有图表生成完成！")


if __name__ == "__main__":
    plot_all()
