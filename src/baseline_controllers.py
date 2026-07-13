#!/usr/bin/env python
"""传统控制方法对比：LQR控制器和简单规则控制器。"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import gymnasium as gym
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def cartpole_dynamics():
    """获取 CartPole 线性化后的系统矩阵 A, B（在平衡点附近）。"""
    # CartPole 物理参数
    m_cart = 1.0    # 小车质量
    m_pole = 0.1    # 摆杆质量
    m_total = m_cart + m_pole
    l = 0.5          # 摆杆半长
    g = 9.81         # 重力加速度

    # 线性化后的状态空间模型: dx/dt = Ax + Bu
    # 状态: [x, x_dot, theta, theta_dot]
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, -m_pole * g / m_cart, 0],
        [0, 0, 0, 1],
        [0, 0, (m_total * g) / (m_cart * l), 0]
    ])

    B = np.array([
        [0],
        [1 / m_cart],
        [0],
        [-1 / (m_cart * l)]
    ])

    return A, B


def solve_lqr(Q, R, A, B, max_iter=1000):
    """通过迭代求解代数Riccati方程获得LQR增益矩阵K。"""
    P = Q.copy()
    for _ in range(max_iter):
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        if np.allclose(P, P_new, atol=1e-8):
            break
        P = P_new
    return K


class LQRController:
    """LQR (Linear Quadratic Regulator) 控制器。"""

    def __init__(self):
        A, B = cartpole_dynamics()
        # 权重矩阵：状态偏差 vs 控制代价
        Q = np.diag([10, 1, 100, 1])  # 重点惩罚角度偏差
        R = np.array([[1]])            # 控制力代价
        self.K = solve_lqr(Q, R, A, B)
        print(f"  LQR 增益矩阵 K: {self.K.flatten()}")

    def get_action(self, obs):
        """根据状态观测返回控制动作。"""
        # obs: [x, x_dot, theta, theta_dot]
        u = -self.K @ obs
        # 将连续控制力映射到离散动作 (左/右)
        # CartPole的动作是施加10N的力，这里用阈值判断
        return 1 if u[0] > 0 else 0


class RuleBasedController:
    """基于规则的简单控制器（角度阈值法）。"""

    def __init__(self, angle_threshold=0.05):
        """
        简单规则：
        - 如果摆杆向右倾斜(theta > threshold)，向右推
        - 如果摆杆向左倾斜(theta < -threshold)，向左推
        - 如果角度很小但有角速度，根据角速度方向推
        """
        self.angle_threshold = angle_threshold

    def get_action(self, obs):
        theta = obs[2]       # 摆杆角度
        theta_dot = obs[3]   # 摆杆角速度

        if theta > self.angle_threshold:
            return 1  # 向右推
        elif theta < -self.angle_threshold:
            return 0  # 向左推
        else:
            # 角度很小时，根据角速度方向推
            return 1 if theta_dot > 0 else 0


def evaluate_controller(controller, n_episodes=10, label=""):
    """评估控制器性能。"""
    env = gym.make("CartPole-v1")

    all_rewards = []
    all_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = controller.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        all_rewards.append(total_reward)
        all_lengths.append(steps)

    env.close()

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)

    print(f"\n  [{label}] {n_episodes} 回合评估结果:")
    print(f"    平均奖励: {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"    平均步数: {np.mean(all_lengths):.1f}")
    print(f"    最高奖励: {max(all_rewards):.0f}")
    print(f"    最低奖励: {min(all_rewards):.0f}")

    return {
        "label": label,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "mean_length": float(np.mean(all_lengths)),
        "rewards": [float(r) for r in all_rewards],
    }


def run_baseline():
    """运行传统控制方法对比实验。"""
    print("\n" + "=" * 60)
    print("  传统控制方法对比")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. 随机策略基线 ---
    print("\n--- 随机策略 ---")

    class RandomController:
        def __init__(self):
            pass
        def get_action(self, obs):
            return np.random.randint(0, 2)

    random_result = evaluate_controller(RandomController(), n_episodes=20, label="随机策略")

    # --- 2. 简单规则控制器 ---
    print("\n--- 简单规则控制器 ---")
    rule_controller = RuleBasedController(angle_threshold=0.05)
    rule_result = evaluate_controller(rule_controller, n_episodes=20, label="规则控制器")

    # --- 3. LQR 控制器 ---
    print("\n--- LQR 控制器 ---")
    lqr_controller = LQRController()
    lqr_result = evaluate_controller(lqr_controller, n_episodes=20, label="LQR控制器")

    # --- 4. 评估训练好的 RL 智能体 ---
    rl_results = {}
    for algo in ["dqn", "ppo"]:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", f"{algo}_cartpole")
        if os.path.exists(model_path + ".zip") or os.path.exists(model_path):
            print(f"\n--- {algo.upper()} 智能体 ---")
            if algo == "dqn":
                from stable_baselines3 import DQN
                model = DQN.load(model_path)
            else:
                from stable_baselines3 import PPO
                model = PPO.load(model_path)

            env = gym.make("CartPole-v1")
            all_rewards = []
            for ep in range(20):
                obs, info = env.reset(seed=ep)
                done = False
                total_reward = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    done = terminated or truncated
                    total_reward += reward
                all_rewards.append(total_reward)
            env.close()

            mean_r = np.mean(all_rewards)
            print(f"  [{algo.upper()}] 20回合平均奖励: {mean_r:.1f}")
            rl_results[algo] = {
                "label": algo.upper(),
                "mean_reward": float(mean_r),
                "rewards": [float(r) for r in all_rewards],
            }

    # --- 汇总对比 ---
    print("\n" + "=" * 60)
    print("  控制方法对比汇总")
    print("=" * 60)
    print(f"  {'方法':<15} {'平均奖励':>10} {'平均步数':>10}")
    print(f"  {'-'*35}")
    print(f"  {'随机策略':<15} {random_result['mean_reward']:>10.1f} {random_result['mean_length']:>10.1f}")
    print(f"  {'规则控制器':<15} {rule_result['mean_reward']:>10.1f} {rule_result['mean_length']:>10.1f}")
    print(f"  {'LQR控制器':<15} {lqr_result['mean_reward']:>10.1f} {lqr_result['mean_length']:>10.1f}")
    for algo, data in rl_results.items():
        print(f"  {data['label']:<15} {data['mean_reward']:>10.1f}")
    print("=" * 60)

    # 保存对比结果
    comparison = {
        "random": random_result,
        "rule_based": rule_result,
        "lqr": lqr_result,
        **rl_results,
    }
    result_path = os.path.join(RESULTS_DIR, "baseline_comparison.json")
    with open(result_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n  对比结果已保存: {result_path}")


if __name__ == "__main__":
    run_baseline()
