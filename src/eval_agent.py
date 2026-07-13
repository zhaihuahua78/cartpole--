#!/usr/bin/env python
"""评估训练好的智能体（DQN/PPO）。"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import gymnasium as gym
import json

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def evaluate(algo="dqn", n_episodes=10, verbose=True):
    """加载模型并评估性能。"""
    model_name = f"{algo}_cartpole"
    model_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}，请先训练")
        return None

    if algo == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(model_path)
    elif algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
    else:
        print(f"不支持的算法: {algo}")
        return None

    env = gym.make("CartPole-v1")

    all_rewards = []
    all_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += reward
            steps += 1

        all_rewards.append(total_reward)
        all_lengths.append(steps)

        if verbose:
            print(f"  Episode {ep+1:2d}: reward={total_reward:6.0f}  steps={steps}")

    env.close()

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_length = np.mean(all_lengths)

    if verbose:
        print(f"\n{'='*50}")
        print(f"  {algo.upper()} 评估结果 ({n_episodes} 回合)")
        print(f"{'='*50}")
        print(f"  平均奖励:  {mean_reward:.1f} ± {std_reward:.1f}")
        print(f"  平均步数:  {mean_length:.1f}")
        print(f"  最高奖励:  {max(all_rewards):.0f}")
        print(f"  最低奖励:  {min(all_rewards):.0f}")
        solved = sum(1 for r in all_rewards if r >= 475)
        print(f"  达标回合:  {solved}/{n_episodes} (奖励>=475)")
        print(f"{'='*50}")

    # 保存评估结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        "algo": algo,
        "n_episodes": n_episodes,
        "rewards": all_rewards,
        "lengths": all_lengths,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
    }
    result_path = os.path.join(RESULTS_DIR, f"{algo}_eval_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  评估结果已保存: {result_path}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    evaluate(args.algo, args.episodes)
