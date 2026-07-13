#!/usr/bin/env python
"""自定义奖励函数对比实验。

对比 CartPole-v1 默认奖励（每步+1）与自定义连续奖励函数的效果差异。
自定义奖励函数综合考虑摆杆角度、角速度、小车位置和速度，提供更密集的反馈信号。
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


class CustomRewardWrapper(gym.Wrapper):
    """
    自定义奖励函数包装器。

    设计思路：
    - 基础奖励：每步存活 +1（保留默认机制）
    - 角度惩罚：|θ|/θ_max 越大，惩罚越大（鼓励保持竖直）
    - 角速度惩罚：|θ̇| 越大，惩罚越大（鼓励减少摆动）
    - 位置惩罚：|x|/x_max 越大，惩罚越大（鼓励保持在中心）

    综合奖励 = 1.0 - α·|θ|/θ_max - β·|θ̇|·w_ω - γ·|x|/x_max
    其中 θ_max=0.2094 rad, x_max=2.4 m
    """

    def __init__(self, env, alpha=0.5, beta=0.1, gamma=0.05, w_omega=0.1):
        super().__init__(env)
        self.alpha = alpha   # 角度权重
        self.beta = beta     # 角速度权重
        self.gamma = gamma   # 位置权重
        self.w_omega = w_omega  # 角速度缩放因子
        self.theta_max = 0.2094  # 终止角度
        self.x_max = 2.4         # 终止位置

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            # 失败时给予负奖励
            custom_reward = -10.0
        else:
            x, x_dot, theta, theta_dot = obs
            # 归一化各状态量
            angle_penalty = abs(theta) / self.theta_max
            omega_penalty = abs(theta_dot) * self.w_omega
            position_penalty = abs(x) / self.x_max

            # 综合奖励：基础分1.0减去各项惩罚
            custom_reward = (1.0
                           - self.alpha * angle_penalty
                           - self.beta * omega_penalty
                           - self.gamma * position_penalty)
            # 确保奖励为正
            custom_reward = max(custom_reward, 0.0)

        info["custom_reward"] = custom_reward
        return obs, custom_reward, terminated, truncated, info


class EpisodeRewardTracker(BaseCallback):
    """记录每回合的原始环境奖励和自定义奖励。"""

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


def make_env(custom_reward=False, alpha=0.5, beta=0.1, gamma=0.05):
    """创建环境。"""
    env = gym.make("CartPole-v1")
    if custom_reward:
        env = CustomRewardWrapper(env, alpha=alpha, beta=beta, gamma=gamma)
    env = Monitor(env)
    return env


def train_and_evaluate(custom_reward=False, label="", timesteps=50000,
                       alpha=0.5, beta=0.1, gamma=0.05):
    """训练并评估一个PPO智能体。"""
    print(f"\n  训练: {label} ...")

    env = DummyVecEnv([lambda: make_env(custom_reward=custom_reward,
                                        alpha=alpha, beta=beta, gamma=gamma)])
    tracker = EpisodeRewardTracker()

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        seed=42,
        device="cpu",
    )

    model.learn(total_timesteps=timesteps, callback=tracker)

    # 评估（用默认环境评估以保持公平）
    eval_env = gym.make("CartPole-v1")
    eval_rewards = []
    for ep in range(20):
        obs, _ = eval_env.reset(seed=ep)
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(int(action))
            done = terminated or truncated
            total_reward += reward
        eval_rewards.append(total_reward)
    eval_env.close()
    env.close()

    mean_eval = np.mean(eval_rewards)
    print(f"  -> 评估均值: {mean_eval:.1f} (训练回合数: {len(tracker.episode_rewards)})")
    return {
        "label": label,
        "train_rewards": [float(r) for r in tracker.episode_rewards],
        "eval_rewards": [float(r) for r in eval_rewards],
        "mean_eval": float(mean_eval),
    }


def run_comparison():
    """运行默认奖励 vs 自定义奖励对比实验。"""
    print("\n" + "=" * 60)
    print("  奖励函数设计对比实验")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}

    # 实验1：默认奖励（每步+1）
    results["default"] = train_and_evaluate(
        custom_reward=False,
        label="默认奖励 (每步+1)",
        timesteps=50000,
    )

    # 实验2：自定义连续奖励（标准权重）
    results["custom_v1"] = train_and_evaluate(
        custom_reward=True,
        alpha=0.5, beta=0.1, gamma=0.05,
        label="自定义奖励 v1 (α=0.5, β=0.1, γ=0.05)",
        timesteps=50000,
    )

    # 实验3：自定义连续奖励（加大角度权重）
    results["custom_v2"] = train_and_evaluate(
        custom_reward=True,
        alpha=0.7, beta=0.15, gamma=0.05,
        label="自定义奖励 v2 (α=0.7, β=0.15, γ=0.05)",
        timesteps=50000,
    )

    # 汇总对比
    print("\n" + "=" * 60)
    print("  奖励函数对比汇总")
    print("=" * 60)
    print(f"  {'奖励类型':<35} {'评估均值':>10}")
    print(f"  {'-'*45}")
    for key, data in results.items():
        print(f"  {data['label']:<35} {data['mean_eval']:>10.1f}")
    print("=" * 60)

    # 保存结果
    save_results = {}
    for key, data in results.items():
        save_results[key] = {
            "label": data["label"],
            "mean_eval": data["mean_eval"],
            "eval_rewards": data["eval_rewards"],
            "train_curve": data["train_rewards"][:500],  # 保存前500回合训练曲线
        }
    result_path = os.path.join(RESULTS_DIR, "reward_comparison.json")
    with open(result_path, "w") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {result_path}")


if __name__ == "__main__":
    run_comparison()
