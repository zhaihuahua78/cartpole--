#!/usr/bin/env python
"""用随机策略演示 CartPole 环境，并保存初始状态截图。"""

import sys
import os

# 强制UTF-8输出，避免Windows GBK编码问题
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，用于服务器环境
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def render_cartpole_frame(obs, step=0):
    """手动绘制 CartPole 状态的简化示意图并保存。"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    x_cart = obs[0]
    theta = obs[2]

    # 小车
    cart_width, cart_height = 0.6, 0.3
    cart_rect = plt.Rectangle(
        (x_cart - cart_width / 2, -cart_height / 2),
        cart_width, cart_height,
        facecolor="steelblue", edgecolor="black", linewidth=2
    )
    ax.add_patch(cart_rect)

    # 车轮
    wheel_r = 0.07
    for cx in [x_cart - cart_width / 4, x_cart + cart_width / 4]:
        wheel = plt.Circle((cx, -cart_height / 2 - wheel_r), wheel_r,
                           facecolor="gray", edgecolor="black", linewidth=1.5)
        ax.add_patch(wheel)

    # 摆杆
    pole_length = 1.0
    x_pole = x_cart
    y_pole = 0.0
    dx = pole_length * np.sin(theta)
    dy = pole_length * np.cos(theta)
    ax.plot([x_pole, x_pole + dx], [y_pole, y_pole + dy],
            color="brown", linewidth=4, solid_capstyle="round")

    # 摆杆顶端
    ax.plot(x_pole + dx, y_pole + dy, "o", color="red", markersize=8)

    # 轨道
    ax.plot([-3, 3], [-cart_height / 2 - 2 * wheel_r, -cart_height / 2 - 2 * wheel_r],
            "k-", linewidth=2)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.8, 1.5)
    ax.set_aspect("equal")
    ax.set_title(f"CartPole  Step={step}  θ={np.degrees(theta):.1f}°  x={x_cart:.3f}m")
    ax.set_xlabel("x (m)")
    ax.grid(True, alpha=0.3)

    return fig


def run_demo(n_steps=200, save_gif=True):
    """用随机策略运行 CartPole，保存状态截图和可选 GIF。"""
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=42)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 保存初始帧
    fig = render_cartpole_frame(obs, step=0)
    fig.savefig(os.path.join(RESULTS_DIR, "env_demo_initial.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("✅ 已保存初始状态截图: results/env_demo_initial.png")

    if save_gif:
        import imageio.v2 as imageio
        frames = []
        frames.append(_fig_to_array(obs, 0))

        total_reward = 0
        for step in range(1, n_steps + 1):
            action = env.action_space.sample()  # 随机动作
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if step % 5 == 0 or done:
                frames.append(_fig_to_array(obs, step))

            if done:
                break

        gif_path = os.path.join(RESULTS_DIR, "env_demo_random.gif")
        imageio.mimsave(gif_path, frames, duration=50)
        print(f"✅ 已保存随机策略演示 GIF: {gif_path}")
        print(f"   随机策略回合奖励: {total_reward} (步数)")
    else:
        total_reward = 0
        for step in range(1, n_steps + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"   随机策略回合奖励: {total_reward}")

    env.close()
    return total_reward


def _fig_to_array(obs, step):
    """将当前状态渲染为 numpy 数组（用于 GIF 帧）。"""
    fig = render_cartpole_frame(obs, step)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3]  # 去掉 alpha
    plt.close(fig)
    return arr


if __name__ == "__main__":
    run_demo()
