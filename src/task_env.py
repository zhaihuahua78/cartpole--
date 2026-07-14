"""
符合任务描述的自定义 CartPole 环境包装器。

修改点：
1. 终止角度：12° → 15°（0.2618 rad），与任务描述一致
2. 失败惩罚：回合终止时给予 -10 惩罚，与任务描述"获得相应的负向惩罚"一致
3. 小车边界 2.4m、最大步数 500 保持不变
4. 每步存活奖励 +1 保持不变

实现方式：直接实现 CartPole 物理引擎，修改终止角度和惩罚值。
这样避免了 Gymnasium 在 12° 终止后无法继续步进的限制。
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*terminated = True.*")


class TaskCartPoleEnv(gym.Env):
    """
    符合任务描述的 CartPole 环境。
    
    与 CartPole-v1 的唯一区别：
    - 终止角度从 12° (0.2094 rad) 改为 15° (0.2618 rad)
    - 失败时（角度或位置超限）给予 -10 惩罚
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, fail_penalty=-10.0, theta_deg=15.0):
        super().__init__()
        # 物理参数（与 CartPole-v1 一致）
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        # 自定义参数
        self.fail_penalty = fail_penalty
        self.theta_threshold_radians = np.radians(theta_deg)
        self.x_threshold = 2.4
        self._step_count = 0
        self._max_steps = 500

        # 状态/动作空间（与 CartPole-v1 一致）
        high = np.array([4.8, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max],
                        dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def step(self, action):
        self._step_count += 1
        x, x_dot, theta, theta_dot = self.state

        # 施加力
        force = self.force_mag if action == 1 else -self.force_mag

        # 物理方程（与 CartPole-v1 完全一致）
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # 更新状态（半隐式欧拉）
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # 判断终止条件
        angle_failed = abs(theta) > self.theta_threshold_radians
        position_failed = abs(x) > self.x_threshold
        task_failed = angle_failed or position_failed
        max_steps_reached = self._step_count >= self._max_steps

        if task_failed:
            # 失败：负向惩罚（任务描述要求）
            reward = self.fail_penalty
            terminated = True
            truncated = False
        elif max_steps_reached:
            # 成功坚持500步：与之前每步一致的+1奖励
            reward = 1.0
            terminated = False
            truncated = True
        else:
            # 存活：+1 奖励
            reward = 1.0
            terminated = False
            truncated = False

        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        # 初始状态：在平衡点附近小范围随机扰动
        x = self.np_random.uniform(low=-0.05, high=0.05)
        x_dot = self.np_random.uniform(low=-0.05, high=0.05)
        theta = self.np_random.uniform(low=-0.05, high=0.05)
        theta_dot = self.np_random.uniform(low=-0.05, high=0.05)
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        return self.state, {}

    def render(self):
        # 可选渲染——暂不实现
        pass

    def close(self):
        pass


def make_task_env(render_mode=None, fail_penalty=-10.0, theta_deg=15.0):
    """创建符合任务描述的 CartPole 环境。"""
    return TaskCartPoleEnv(
        render_mode=render_mode,
        fail_penalty=fail_penalty,
        theta_deg=theta_deg,
    )


if __name__ == "__main__":
    env = make_task_env()
    obs, info = env.reset(seed=42)
    theta_rad = np.radians(15.0)
    print(f"终止角度阈值: {theta_rad:.4f} rad (15.0°)")
    print(f"失败惩罚: {env.fail_penalty}")
    print(f"初始观测: {obs}")
    print(f"最大步数: 500")

    # 测试随机策略
    total_reward = 0
    done = False
    step = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        if done:
            theta_deg_val = np.degrees(abs(obs[2]))
            x_pos = abs(obs[0])
            reason = "angle" if abs(obs[2]) > env.theta_threshold_radians else "position" if abs(obs[0]) > env.x_threshold else f"step_{step}"
            print(f"回合结束: step={step}, total_reward={total_reward:.0f}, 原因={reason}, |θ|={theta_deg_val:.1f}°, |x|={x_pos:.2f}m")
    env.close()
