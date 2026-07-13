#!/usr/bin/env python
"""PPO 算法训练 CartPole-v1。"""

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

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


class EpisodeRewardTracker(BaseCallback):
    """记录每个回合的奖励和步数。"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        self._current_length += 1

        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            ep = len(self.episode_rewards)
            if ep % 10 == 0 or self.episode_rewards[-1] >= 475:
                recent = self.episode_rewards[-10:]
                print(f"  Episode {ep:4d} | Reward: {self.episode_rewards[-1]:6.0f} | "
                      f"Mean(last 10): {np.mean(recent):6.1f} | Length: {self.episode_lengths[-1]}")
            self._current_reward = 0.0
            self._current_length = 0
        return True

    def save(self, path):
        data = {
            "episode_rewards": [float(r) for r in self.episode_rewards],
            "episode_lengths": [int(l) for l in self.episode_lengths],
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"  训练日志已保存: {path}")


def train_ppo(timesteps=50000, learning_rate=3e-4, seed=42):
    """训练 PPO 智能体。"""
    print("\n" + "=" * 60)
    print("  PPO 训练 — CartPole-v1")
    print(f"  总训练步数: {timesteps}, 学习率: {learning_rate}")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 创建环境
    env = DummyVecEnv([lambda: Monitor(gym.make("CartPole-v1"))])

    # 创建回调
    tracker = EpisodeRewardTracker(verbose=0)

    # 创建 PPO 模型
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=seed,
        device="cpu",
    )

    print("\n开始训练...")
    model.learn(total_timesteps=timesteps, callback=tracker)

    # 保存模型
    model_path = os.path.join(MODELS_DIR, "ppo_cartpole")
    model.save(model_path)
    print(f"\n模型已保存: {model_path}")

    # 保存训练日志
    log_path = os.path.join(RESULTS_DIR, "ppo_train_log.json")
    tracker.save(log_path)

    # 汇总
    rewards = tracker.episode_rewards
    if rewards:
        print(f"\n训练汇总:")
        print(f"  总回合数: {len(rewards)}")
        print(f"  最后100回合平均奖励: {np.mean(rewards[-100:]):.1f}")
        print(f"  最高回合奖励: {max(rewards):.0f}")
        if len(rewards) >= 100:
            last100 = rewards[-100:]
            if np.mean(last100) >= 475:
                print("  ✅ CartPole-v1 已解决！(last 100 mean >= 475)")
            else:
                print(f"  ⚠️  未解决 (last 100 mean = {np.mean(last100):.1f} < 475)")

    env.close()
    return model, tracker


if __name__ == "__main__":
    train_ppo()
