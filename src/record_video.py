#!/usr/bin/env python
"""录制训练前后智能体运行的 GIF 视频。"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import gymnasium as gym
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _render_frame(obs, step=0, title=""):
    """将 CartPole 状态渲染为 numpy 数组。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    x_cart = obs[0]
    theta = obs[2]

    # 小车
    cart_w, cart_h = 0.6, 0.3
    cart_rect = plt.Rectangle(
        (x_cart - cart_w / 2, -cart_h / 2), cart_w, cart_h,
        facecolor="steelblue", edgecolor="black", linewidth=2
    )
    ax.add_patch(cart_rect)

    # 车轮
    for cx in [x_cart - cart_w / 4, x_cart + cart_w / 4]:
        wheel = plt.Circle((cx, -cart_h / 2 - 0.07), 0.07,
                           facecolor="gray", edgecolor="black", linewidth=1.5)
        ax.add_patch(wheel)

    # 摆杆
    pole_len = 1.0
    dx = pole_len * np.sin(theta)
    dy = pole_len * np.cos(theta)
    ax.plot([x_cart, x_cart + dx], [0, dy],
            color="brown", linewidth=4, solid_capstyle="round")
    ax.plot(x_cart + dx, dy, "o", color="red", markersize=6)

    # 轨道
    ax.plot([-3, 3], [-0.44, -0.44], "k-", linewidth=2)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.8, 1.5)
    ax.set_aspect("equal")
    ax.set_title(f"{title}  Step={step}  Reward={step+1}")
    ax.grid(True, alpha=0.2)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return arr


def record(algo="ppo", seed=42):
    """录制智能体运行视频（GIF格式）。"""
    import imageio.v2 as imageio

    model_path = os.path.join(MODELS_DIR, f"{algo}_cartpole")
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"⚠️  模型不存在: {model_path}，请先训练")
        return

    if algo == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(model_path)
    else:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)

    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=seed)

    frames = []
    frames.append(_render_frame(obs, step=0, title=f"{algo.upper()} Trained Agent"))

    total_reward = 0
    step = 0
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        total_reward += reward
        step += 1

        # 每隔几帧录制一帧（减小文件大小）
        if step % 2 == 0 or done:
            frames.append(_render_frame(obs, step=step, title=f"{algo.upper()} Trained Agent"))

    env.close()

    # 保存 GIF
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gif_path = os.path.join(RESULTS_DIR, f"{algo}_demo.gif")
    imageio.mimsave(gif_path, frames, duration=50)
    print(f"✅ 已录制 {algo.upper()} 演示视频: {gif_path}")
    print(f"   总奖励: {total_reward}, 总步数: {step}")

    # 保存关键帧截图
    frame_indices = [0, len(frames) // 2, len(frames) - 1]
    labels = ["initial", "mid", "final"]
    for fi, label in zip(frame_indices, labels):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.imshow(frames[fi])
        ax.axis("off")
        save_name = os.path.join(RESULTS_DIR, f"{algo}_frame_{label}.png")
        plt.savefig(save_name, dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"   关键帧截图已保存")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="ppo")
    args = parser.parse_args()
    record(args.algo)
