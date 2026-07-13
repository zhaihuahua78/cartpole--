#!/usr/bin/env python
"""打印 CartPole-v1 环境的状态空间与动作空间信息。"""

import sys
import os

# 强制UTF-8输出，避免Windows GBK编码问题
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import numpy as np


def print_env_info():
    """打印 CartPole-v1 环境的详细信息。"""
    env = gym.make("CartPole-v1")

    print("=" * 60)
    print("  CartPole-v1 环境信息")
    print("=" * 60)

    # --- 状态空间 (Observation Space) ---
    obs_space = env.observation_space
    print(f"\n【状态空间 (Observation Space)】")
    print(f"  类型: {obs_space}")
    print(f"  维度: {obs_space.shape}")
    print(f"  数据类型: {obs_space.dtype}")

    state_names = ["小车位置 (x)", "小车速度 (ẋ)", "摆杆角度 (θ)", "摆杆角速度 (θ̇)"]
    state_units = ["m", "m/s", "rad", "rad/s"]
    print(f"\n  各状态量含义:")
    for i, (name, unit) in enumerate(zip(state_names, state_units)):
        low = obs_space.low[i]
        high = obs_space.high[i]
        low_str = f"{low:.4f}" if abs(low) < 1e6 else "未限制"
        high_str = f"{high:.4f}" if abs(high) < 1e6 else "未限制"
        print(f"    [{i}] {name}: 范围 [{low_str}, {high_str}] {unit}")

    # --- 动作空间 (Action Space) ---
    act_space = env.action_space
    print(f"\n【动作空间 (Action Space)】")
    print(f"  类型: {act_space}")
    print(f"  动作数: {act_space.n}")
    print(f"    0 — 向左推力 (左移)")
    print(f"    1 — 向右推力 (右移)")

    # --- 终止条件 ---
    print(f"\n【终止条件 (v1)】")
    print(f"  - 摆杆角度 |θ| > 12° (0.2094 rad)")
    print(f"  - 小车位置 |x| > 2.4 m (相对于中心)")
    print(f"  - 回合步数达到 500 步")

    # --- 奖励机制 ---
    print(f"\n【奖励机制】")
    print(f"  - 每步存活: +1")
    print(f"  - 一回合最高奖励: 500")

    # --- 演示一次初始状态 ---
    print(f"\n【初始状态演示】")
    obs, info = env.reset(seed=42)
    print(f"  初始观测: {obs}")
    print(f"    小车位置: {obs[0]:.4f} m")
    print(f"    小车速度: {obs[1]:.4f} m/s")
    print(f"    摆杆角度: {obs[2]:.4f} rad ({np.degrees(obs[2]):.2f}°)")
    print(f"    摆杆角速度: {obs[3]:.4f} rad/s")

    env.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_env_info()
