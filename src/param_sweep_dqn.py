#!/usr/bin/env python
"""DQN 关键参数搜索与对比。"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import json
import copy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


class EpisodeRewardTracker(BaseCallback):
    """记录每回合奖励。"""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self):
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True


def _train_single_dqn(params, total_timesteps=30000, seed=42):
    """用给定参数训练一个 DQN，返回每回合奖励列表。"""
    env = DummyVecEnv([lambda: Monitor(gym.make("CartPole-v1"))])
    tracker = EpisodeRewardTracker()

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=params.get("learning_rate", 1e-4),
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=params.get("exploration_fraction", 0.1),
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=0,
        seed=seed,
        device="cpu",
    )

    model.learn(total_timesteps=total_timesteps, callback=tracker)
    env.close()
    return tracker.episode_rewards


def sweep_dqn():
    """对 DQN 的关键参数进行对比实验。"""
    print("\n" + "=" * 60)
    print("  DQN 参数搜索")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 实验1: 学习率 ---
    print("\n【实验1: 学习率对比】")
    lr_configs = [
        {"learning_rate": 1e-5, "label": "lr=1e-5"},
        {"learning_rate": 1e-4, "label": "lr=1e-4 (默认)"},
        {"learning_rate": 3e-4, "label": "lr=3e-4"},
    ]
    lr_results = {}
    for cfg in lr_configs:
        print(f"\n  训练: {cfg['label']} ...")
        rewards = _train_single_dqn(cfg, total_timesteps=30000)
        lr_results[cfg["label"]] = rewards
        last50 = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        print(f"  -> 最后50回合平均: {last50:.1f}")

    # --- 实验2: 探索比例 ---
    print("\n【实验2: 探索比例对比】")
    exp_configs = [
        {"exploration_fraction": 0.05, "label": "exp_frac=0.05"},
        {"exploration_fraction": 0.1, "label": "exp_frac=0.1 (默认)"},
        {"exploration_fraction": 0.2, "label": "exp_frac=0.2"},
    ]
    exp_results = {}
    for cfg in exp_configs:
        print(f"\n  训练: {cfg['label']} ...")
        rewards = _train_single_dqn(cfg, total_timesteps=30000)
        exp_results[cfg["label"]] = rewards
        last50 = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        print(f"  -> 最后50回合平均: {last50:.1f}")

    # 保存结果
    all_results = {
        "learning_rate_sweep": lr_results,
        "exploration_sweep": exp_results,
    }
    # 序列化时把numpy list转成普通list
    save_results = {}
    for key, sub in all_results.items():
        save_results[key] = {k: [float(v) for v in vals] for k, vals in sub.items()}

    result_path = os.path.join(RESULTS_DIR, "dqn_sweep_results.json")
    with open(result_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  参数搜索结果已保存: {result_path}")


if __name__ == "__main__":
    sweep_dqn()
