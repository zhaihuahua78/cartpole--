#!/usr/bin/env python
"""PPO 关键参数搜索与对比。"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import json

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


def _train_single_ppo(params, total_timesteps=30000, seed=42):
    """用给定参数训练一个 PPO，返回每回合奖励列表。"""
    env = DummyVecEnv([lambda: Monitor(gym.make("CartPole-v1"))])
    tracker = EpisodeRewardTracker()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=params.get("learning_rate", 3e-4),
        n_steps=2048,
        batch_size=64,
        n_epochs=params.get("n_epochs", 10),
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=params.get("clip_range", 0.2),
        ent_coef=0.0,
        vf_coef=0.5,
        verbose=0,
        seed=seed,
        device="cpu",
    )

    model.learn(total_timesteps=total_timesteps, callback=tracker)
    env.close()
    return tracker.episode_rewards


def sweep_ppo():
    """对 PPO 的关键参数进行对比实验。"""
    print("\n" + "=" * 60)
    print("  PPO 参数搜索")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 实验1: 学习率 ---
    print("\n【实验1: 学习率对比】")
    lr_configs = [
        {"learning_rate": 1e-4, "label": "lr=1e-4"},
        {"learning_rate": 3e-4, "label": "lr=3e-4 (默认)"},
        {"learning_rate": 1e-3, "label": "lr=1e-3"},
    ]
    lr_results = {}
    for cfg in lr_configs:
        print(f"\n  训练: {cfg['label']} ...")
        rewards = _train_single_ppo(cfg, total_timesteps=30000)
        lr_results[cfg["label"]] = rewards
        last50 = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        print(f"  -> 最后50回合平均: {last50:.1f}")

    # --- 实验2: clip_range ---
    print("\n【实验2: clip_range 对比】")
    clip_configs = [
        {"clip_range": 0.1, "label": "clip=0.1"},
        {"clip_range": 0.2, "label": "clip=0.2 (默认)"},
        {"clip_range": 0.3, "label": "clip=0.3"},
    ]
    clip_results = {}
    for cfg in clip_configs:
        print(f"\n  训练: {cfg['label']} ...")
        rewards = _train_single_ppo(cfg, total_timesteps=30000)
        clip_results[cfg["label"]] = rewards
        last50 = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        print(f"  -> 最后50回合平均: {last50:.1f}")

    # 保存结果
    all_results = {
        "learning_rate_sweep": lr_results,
        "clip_range_sweep": clip_results,
    }
    save_results = {}
    for key, sub in all_results.items():
        save_results[key] = {k: [float(v) for v in vals] for k, vals in sub.items()}

    result_path = os.path.join(RESULTS_DIR, "ppo_sweep_results.json")
    with open(result_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  参数搜索结果已保存: {result_path}")


if __name__ == "__main__":
    sweep_ppo()
