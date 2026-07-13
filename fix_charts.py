#!/usr/bin/env python
"""重新绘制所有图表 - 修复中文显示问题"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 关键：强制使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 清除字体缓存
fm._load_fontmanager(try_read_cache=False)

# 颜色主题
COLORS = {
    'dqn': '#2E86AB',
    'ppo': '#A23B72',
    'lqr': '#F18F01',
    'rule': '#C73E1D',
    'random': '#3B1F2B',
    'bg': '#FAFAFA',
    'grid': '#E8E8E8',
    'text': '#2C3E50',
    'success': '#27AE60',
}

def smooth_data(data, window=10):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_training_curves():
    """DQN与PPO训练曲线对比"""
    with open('results/dqn_train_log.json') as f:
        dqn_data = json.load(f)
    with open('results/ppo_train_log.json') as f:
        ppo_data = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS['bg'])
    
    # DQN
    ax1 = axes[0]
    ax1.set_facecolor(COLORS['bg'])
    dqn_rewards = dqn_data['episode_rewards']
    ax1.plot(range(1, len(dqn_rewards)+1), dqn_rewards, 
             color=COLORS['dqn'], alpha=0.15, linewidth=0.8)
    if len(dqn_rewards) > 20:
        smoothed = smooth_data(dqn_rewards, window=20)
        ax1.plot(range(20, len(dqn_rewards)+1), smoothed, 
                 color=COLORS['dqn'], linewidth=2.5, label='Moving Avg (w=20)')
    ax1.axhline(y=475, color=COLORS['success'], linestyle='--', linewidth=1.5, alpha=0.7, label='Solved (475)')
    ax1.axhline(y=500, color=COLORS['success'], linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Episodes', fontsize=11, color=COLORS['text'])
    ax1.set_ylabel('Reward', fontsize=11, color=COLORS['text'])
    ax1.set_title('DQN Training Curve', fontsize=13, fontweight='bold', color=COLORS['text'], pad=10)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors=COLORS['text'])
    
    # PPO
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['bg'])
    ppo_rewards = ppo_data['episode_rewards']
    ax2.plot(range(1, len(ppo_rewards)+1), ppo_rewards, 
             color=COLORS['ppo'], alpha=0.15, linewidth=0.8)
    if len(ppo_rewards) > 10:
        smoothed_ppo = smooth_data(ppo_rewards, window=10)
        ax2.plot(range(10, len(ppo_rewards)+1), smoothed_ppo, 
                 color=COLORS['ppo'], linewidth=2.5, label='Moving Avg (w=10)')
    ax2.axhline(y=475, color=COLORS['success'], linestyle='--', linewidth=1.5, alpha=0.7, label='Solved (475)')
    ax2.axhline(y=500, color=COLORS['success'], linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Episodes', fontsize=11, color=COLORS['text'])
    ax2.set_ylabel('Reward', fontsize=11, color=COLORS['text'])
    ax2.set_title('PPO Training Curve', fontsize=13, fontweight='bold', color=COLORS['text'], pad=10)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors=COLORS['text'])
    
    plt.tight_layout(pad=2.0)
    plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print('[OK] training_curves.png')


def plot_baseline_comparison():
    """控制方法对比"""
    with open('results/baseline_comparison.json') as f:
        data = json.load(f)
    
    method_map = {
        'random': ('Random', COLORS['random']),
        'rule_based': ('Rule-based', COLORS['rule']),
        'lqr': ('LQR', COLORS['lqr']),
        'dqn': ('DQN', COLORS['dqn']),
        'ppo': ('PPO', COLORS['ppo']),
    }
    
    methods, rewards, colors, errors = [], [], [], []
    for key in ['random', 'rule_based', 'lqr', 'dqn', 'ppo']:
        if key in data:
            methods.append(method_map[key][0])
            rewards.append(data[key]['mean_reward'])
            colors.append(method_map[key][1])
            errors.append(data[key].get('std_reward', 0))
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    x = np.arange(len(methods))
    bars = ax.bar(x, rewards, 0.6, color=colors, edgecolor='white', linewidth=1.5, zorder=3)
    
    if any(e > 0 for e in errors):
        ax.errorbar(x, rewards, yerr=errors, fmt='none', color=COLORS['text'], capsize=5, capthick=1.5, linewidth=1.5)
    
    for bar, reward in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 8,
                f'{reward:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=COLORS['text'])
    
    ax.axhline(y=475, color=COLORS['success'], linestyle='--', linewidth=1.5, alpha=0.7, label='Solved (475)')
    ax.set_xlabel('Control Method', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('Average Reward', fontsize=12, color=COLORS['text'])
    ax.set_title('Control Method Performance Comparison', fontsize=14, fontweight='bold', color=COLORS['text'], pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, color=COLORS['text'])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(bottom=0, top=max(rewards) * 1.15)
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.tick_params(colors=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig('results/baseline_comparison.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print('[OK] baseline_comparison.png')


def plot_before_after():
    """训练前后对比"""
    import gymnasium as gym
    
    env = gym.make("CartPole-v1")
    pre_rewards = []
    for ep in range(20):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        pre_rewards.append(total_reward)
    env.close()
    
    from stable_baselines3 import PPO, DQN
    
    post_rewards_ppo = []
    model_path = 'models/ppo_cartpole'
    if os.path.exists(model_path + '.zip') or os.path.exists(model_path):
        model = PPO.load(model_path)
        env = gym.make("CartPole-v1")
        for ep in range(20):
            obs, _ = env.reset(seed=ep)
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
                total_reward += reward
            post_rewards_ppo.append(total_reward)
        env.close()
    
    post_rewards_dqn = []
    model_path = 'models/dqn_cartpole'
    if os.path.exists(model_path + '.zip') or os.path.exists(model_path):
        model = DQN.load(model_path)
        env = gym.make("CartPole-v1")
        for ep in range(20):
            obs, _ = env.reset(seed=ep)
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
                total_reward += reward
            post_rewards_dqn.append(total_reward)
        env.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=COLORS['bg'])
    
    # Bar chart
    ax1 = axes[0]
    ax1.set_facecolor(COLORS['bg'])
    
    categories = ['Before\n(Random)']
    values = [np.mean(pre_rewards)]
    bar_colors = [COLORS['random']]
    
    if post_rewards_dqn:
        categories.append('After\n(DQN)')
        values.append(np.mean(post_rewards_dqn))
        bar_colors.append(COLORS['dqn'])
    if post_rewards_ppo:
        categories.append('After\n(PPO)')
        values.append(np.mean(post_rewards_ppo))
        bar_colors.append(COLORS['ppo'])
    
    x = np.arange(len(categories))
    bars = ax1.bar(x, values, 0.5, color=bar_colors, edgecolor='white', linewidth=1.5, zorder=3)
    
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 8,
                f'{val:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color=COLORS['text'])
    
    ax1.set_ylabel('Average Reward', fontsize=11, color=COLORS['text'])
    ax1.set_title('Before vs After Training', fontsize=13, fontweight='bold', color=COLORS['text'], pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10, color=COLORS['text'])
    ax1.grid(True, axis='y', alpha=0.3, zorder=0)
    ax1.set_ylim(bottom=0)
    ax1.tick_params(colors=COLORS['text'])
    
    # Box plot
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['bg'])
    
    box_data = [pre_rewards]
    box_labels = ['Before']
    box_colors = [COLORS['random']]
    
    if post_rewards_dqn:
        box_data.append(post_rewards_dqn)
        box_labels.append('DQN')
        box_colors.append(COLORS['dqn'])
    if post_rewards_ppo:
        box_data.append(post_rewards_ppo)
        box_labels.append('PPO')
        box_colors.append(COLORS['ppo'])
    
    bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                     medianprops=dict(color='white', linewidth=2))
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Episode Reward', fontsize=11, color=COLORS['text'])
    ax2.set_title('Reward Distribution', fontsize=13, fontweight='bold', color=COLORS['text'], pad=10)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.tick_params(colors=COLORS['text'])
    
    plt.tight_layout(pad=2.0)
    plt.savefig('results/before_after_comparison.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print('[OK] before_after_comparison.png')


if __name__ == '__main__':
    print('Regenerating charts with fixed fonts...\n')
    plot_training_curves()
    plot_baseline_comparison()
    plot_before_after()
    print('\nAll charts regenerated!')
