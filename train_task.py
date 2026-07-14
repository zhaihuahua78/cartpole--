"""
符合任务描述的 CartPole 环境下的完整训练和评估流程。

使用自定义环境：
- 终止角度：15°（任务描述要求）
- 失败惩罚：-10（任务描述要求）
- 每步存活奖励：+1
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import json

# 引入自定义环境
sys.path.insert(0, os.path.dirname(__file__))
from src.task_env import make_task_env

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_task")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models_task")


class EpisodeRewardTracker(BaseCallback):
    """记录每回合奖励。"""

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
            if ep % 20 == 0:
                recent = self.episode_rewards[-20:]
                print(f"  Episode {ep:4d} | Reward: {self._current_reward:6.0f} | "
                      f"Mean(last 20): {np.mean(recent):6.1f} | Length: {self._current_length}")
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


def make_vec_env(seed=42):
    """创建向量化的自定义环境。"""
    def _init():
        return make_task_env()
    return DummyVecEnv([_init])


def train_ppo(timesteps=100000, seed=42):
    """在自定义环境中训练 PPO。"""
    print("\n" + "=" * 60)
    print("  PPO 训练 — 自定义 CartPole (15°, 失败惩罚-10)")
    print(f"  总训练步数: {timesteps}")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = make_vec_env(seed=seed)
    tracker = EpisodeRewardTracker(verbose=0)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        seed=seed,
        device="cpu",
    )

    print("\n开始训练...")
    model.learn(total_timesteps=timesteps, callback=tracker)

    model_path = os.path.join(MODELS_DIR, "ppo_cartpole_task")
    model.save(model_path)
    print(f"\n模型已保存: {model_path}")

    log_path = os.path.join(RESULTS_DIR, "ppo_train_log.json")
    tracker.save(log_path)

    rewards = tracker.episode_rewards
    if rewards:
        print(f"\n训练汇总:")
        print(f"  总回合数: {len(rewards)}")
        print(f"  最后100回合平均奖励: {np.mean(rewards[-100:]):.1f}")
        print(f"  最高回合奖励: {max(rewards):.0f}")

    env.close()
    return model, tracker


def train_dqn(timesteps=300000, seed=42):
    """在自定义环境中训练 DQN。"""
    print("\n" + "=" * 60)
    print("  DQN 训练 — 自定义 CartPole (15°, 失败惩罚-10)")
    print(f"  总训练步数: {timesteps}")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = make_vec_env(seed=seed)
    tracker = EpisodeRewardTracker(verbose=0)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=0,
        seed=seed,
        device="cpu",
    )

    print("\n开始训练...")
    model.learn(total_timesteps=timesteps, callback=tracker)

    model_path = os.path.join(MODELS_DIR, "dqn_cartpole_task")
    model.save(model_path)
    print(f"\n模型已保存: {model_path}")

    log_path = os.path.join(RESULTS_DIR, "dqn_train_log.json")
    tracker.save(log_path)

    rewards = tracker.episode_rewards
    if rewards:
        print(f"\n训练汇总:")
        print(f"  总回合数: {len(rewards)}")
        print(f"  最后100回合平均奖励: {np.mean(rewards[-100:]):.1f}")
        print(f"  最高回合奖励: {max(rewards):.0f}")

    env.close()
    return model, tracker


def evaluate(algo="ppo", n_episodes=20):
    """在自定义环境中评估训练好的模型。"""
    model_name = f"{algo}_cartpole_task"
    model_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        return None

    if algo == "dqn":
        from stable_baselines3 import DQN as ModelClass
    else:
        from stable_baselines3 import PPO as ModelClass

    model = ModelClass.load(model_path)

    all_rewards = []
    all_lengths = []

    for ep in range(n_episodes):
        env = make_task_env()
        obs, info = env.reset(seed=ep)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += reward
            steps += 1

        all_rewards.append(total_reward)
        all_lengths.append(steps)
        print(f"  Episode {ep+1:2d}: reward={total_reward:6.0f}  steps={steps}")

        env.close()

    mean_r = np.mean(all_rewards)
    std_r = np.std(all_rewards)
    solved = sum(1 for r in all_rewards if r >= 475)

    print(f"\n{'='*50}")
    print(f"  {algo.upper()} 评估结果 ({n_episodes} 回合)")
    print(f"{'='*50}")
    print(f"  平均奖励: {mean_r:.1f} ± {std_r:.1f}")
    print(f"  最高奖励: {max(all_rewards):.0f}")
    print(f"  最低奖励: {min(all_rewards):.0f}")
    print(f"  达标回合: {solved}/{n_episodes} (奖励>=475)")
    print(f"{'='*50}")

    return {
        "algo": algo,
        "mean_reward": float(mean_r),
        "std_reward": float(std_r),
        "rewards": all_rewards,
        "lengths": all_lengths,
    }


def run_baseline(n_episodes=20):
    """在自定义环境中运行传统控制基线。"""
    print("\n" + "=" * 60)
    print("  传统控制基线 — 自定义 CartPole (15°, 失败惩罚-10)")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}

    # --- 随机策略 ---
    class RandomController:
        def get_action(self, obs):
            return np.random.randint(0, 2)

    print("\n--- 随机策略 ---")
    random_rewards = []
    for ep in range(n_episodes):
        env = make_task_env()
        obs, _ = env.reset(seed=ep)
        done = False
        total = 0
        while not done:
            action = RandomController().get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        random_rewards.append(total)
        env.close()
    mean_r = np.mean(random_rewards)
    print(f"  平均奖励: {mean_r:.1f}")
    results["random"] = {"mean_reward": float(mean_r), "rewards": random_rewards}

    # --- 规则控制器 ---
    class RuleController:
        def get_action(self, obs):
            theta = obs[2]
            theta_dot = obs[3]
            if theta > 0.05:
                return 1
            elif theta < -0.05:
                return 0
            else:
                return 1 if theta_dot > 0 else 0

    print("\n--- 规则控制器 ---")
    rule_rewards = []
    for ep in range(n_episodes):
        env = make_task_env()
        obs, _ = env.reset(seed=ep)
        done = False
        total = 0
        controller = RuleController()
        while not done:
            action = controller.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        rule_rewards.append(total)
        env.close()
    mean_r = np.mean(rule_rewards)
    print(f"  平均奖励: {mean_r:.1f}")
    results["rule_based"] = {"mean_reward": float(mean_r), "rewards": rule_rewards}

    # --- LQR 控制器 ---
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -0.981, 0],
                  [0, 0, 0, 1],
                  [0, 0, 21.582, 0]])
    B = np.array([[0], [1], [0], [-1.818]])
    Q = np.diag([10, 1, 100, 1])
    R = np.array([[1]])
    P = Q.copy()
    for _ in range(1000):
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        if np.allclose(P, P_new, atol=1e-8):
            break
        P = P_new

    class LQRController:
        def get_action(self, obs):
            u = -K @ obs
            return 1 if u[0] > 0 else 0

    print("\n--- LQR 控制器 ---")
    lqr_rewards = []
    for ep in range(n_episodes):
        env = make_task_env()
        obs, _ = env.reset(seed=ep)
        done = False
        total = 0
        controller = LQRController()
        while not done:
            action = controller.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        lqr_rewards.append(total)
        env.close()
    mean_r = np.mean(lqr_rewards)
    print(f"  平均奖励: {mean_r:.1f}")
    results["lqr"] = {"mean_reward": float(mean_r), "rewards": lqr_rewards}

    # 汇总
    print("\n" + "=" * 60)
    print("  传统控制方法汇总")
    print("=" * 60)
    for key, data in results.items():
        print(f"  {key:<15} {data['mean_reward']:>10.1f}")

    result_path = os.path.join(RESULTS_DIR, "baseline_comparison.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {result_path}")

    return results


def plot_results():
    """生成自定义环境的图表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "ppo_train_log.json")) as f:
        ppo_data = json.load(f)
    with open(os.path.join(RESULTS_DIR, "dqn_train_log.json")) as f:
        dqn_data = json.load(f)
    with open(os.path.join(RESULTS_DIR, "baseline_comparison.json")) as f:
        baseline = json.load(f)

    # --- 训练曲线 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#FAFAFA")

    for idx, (name, color, data) in enumerate(zip(
        ["DQN", "PPO"],
        ["#2E86AB", "#A23B72"],
        [dqn_data, ppo_data]
    )):
        ax = axes[idx]
        ax.set_facecolor("#FAFAFA")
        rewards = data["episode_rewards"]
        ax.plot(range(1, len(rewards)+1), rewards, alpha=0.15, linewidth=0.8, color=color)
        if len(rewards) > 20:
            kernel = np.ones(20) / 20
            smoothed = np.convolve(rewards, kernel, mode="valid")
            ax.plot(range(20, len(rewards)+1), smoothed, color=color, linewidth=2.5, label="Moving Avg (w=20)")
        ax.axhline(y=475, color="green", linestyle="--", alpha=0.5, label="Solved (475+)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title(f"{name} Training Curve (15 deg, penalty=-10)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=min(rewards)-20)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] training_curves.png")

    # --- 对比柱状图 ---
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # RL 评估
    ppo_eval = evaluate("ppo", n_episodes=20)
    dqn_eval = evaluate("dqn", n_episodes=20)

    labels = ["Random", "Rule-based", "LQR", "DQN", "PPO"]
    means = [
        baseline["random"]["mean_reward"],
        baseline["rule_based"]["mean_reward"],
        baseline["lqr"]["mean_reward"],
        dqn_eval["mean_reward"],
        ppo_eval["mean_reward"],
    ]
    colors = ["gray", "orange", "purple", "#2E86AB", "#A23B72"]

    bars = ax.bar(labels, means, color=colors, edgecolor="black", alpha=0.8)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, f"{m:.0f}",
                ha="center", va="bottom", fontweight="bold")
    ax.axhline(y=475, color="red", linestyle="--", alpha=0.5, label="满分阈值 (475)")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Control Method Comparison (15 deg, penalty=-10)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] comparison.png")

    # 保存RL评估结果
    eval_data = {"ppo": ppo_eval, "dqn": dqn_eval}
    with open(os.path.join(RESULTS_DIR, "rl_eval.json"), "w") as f:
        json.dump(eval_data, f, indent=2)
    print("[OK] 评估结果已保存")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", choices=["ppo", "dqn", "both", "none"], default="both")
    parser.add_argument("--baseline", action="store_true", default=True)
    parser.add_argument("--plot", action="store_true", default=True)
    args = parser.parse_args()

    if args.train in ("ppo", "both"):
        train_ppo(timesteps=100000)
    if args.train in ("dqn", "both"):
        train_dqn(timesteps=300000)
    if args.baseline:
        run_baseline()
    if args.plot:
        plot_results()

    print("\n✅ 全部完成！")
