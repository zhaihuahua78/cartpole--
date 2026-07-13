#!/usr/bin/env python
"""重新绘制DQN和PPO训练曲线 - 展示学习过程"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COLORS = {
    'dqn': '#2E86AB',
    'ppo': '#A23B72',
    'bg': '#FAFAFA',
    'success': '#27AE60',
}

def smooth_data(data, window=20):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

def plot_training_curve(algo_name, color, rewards, save_path):
    """绘制单个算法的训练曲线"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    # 原始数据
    episodes = range(1, len(rewards)+1)
    ax.plot(episodes, rewards, color=color, alpha=0.15, linewidth=0.8)
    
    # 平滑曲线
    window = 20 if len(rewards) > 20 else 5
    smoothed = smooth_data(rewards, window=window)
    ax.plot(range(window, len(rewards)+1), smoothed, 
            color=color, linewidth=2.5, label=f'Moving Average (w={window})')
    
    # 满分区域
    ax.axhline(y=475, color=COLORS['success'], linestyle='--', 
               linewidth=1.5, alpha=0.7, label='Solved Threshold (475)')
    ax.axhspan(475, 500, alpha=0.1, color=COLORS['success'])
    
    # 标注满分点
    for i, r in enumerate(rewards):
        if r >= 475:
            ax.scatter(i+1, r, color=COLORS['success'], s=30, zorder=5)
    
    # 统计信息
    max_reward = max(rewards)
    solved_count = sum(1 for r in rewards if r >= 475)
    last100_mean = sum(rewards[-100:])/min(100, len(rewards))
    
    # 注释最大值
    ax.annotate(f'Max: {max_reward:.0f}', 
                xy=(len(rewards), max_reward), 
                xytext=(len(rewards)*0.8, max_reward-80),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, color='black')
    
    # 最后100轮均值线
    ax.axhline(y=last100_mean, color='orange', linestyle=':', 
               linewidth=1, alpha=0.8, label=f'Last 100 Mean: {last100_mean:.0f}')
    
    ax.set_xlabel('Training Episodes', fontsize=12, color='#2C3E50')
    ax.set_ylabel('Episode Reward', fontsize=12, color='#2C3E50')
    ax.set_title(f'{algo_name} Training Curve', fontsize=14, 
                 fontweight='bold', color='#2C3E50', pad=15)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.set_ylim(bottom=0, top=550)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors='#2C3E50')
    
    # 统计文本框
    textstr = f'Episodes: {len(rewards)}\nMax Reward: {max_reward:.0f}\nSolved: {solved_count}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f'[OK] {save_path}')

def plot_combined_curves():
    """绘制DQN和PPO组合图"""
    with open('results/dqn_train_log.json') as f:
        dqn_rewards = json.load(f)['episode_rewards']
    with open('results/ppo_train_log.json') as f:
        ppo_rewards = json.load(f)['episode_rewards']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS['bg'])
    
    for ax, (name, color, rewards) in zip(axes, [
        ('DQN', COLORS['dqn'], dqn_rewards),
        ('PPO', COLORS['ppo'], ppo_rewards)
    ]):
        ax.set_facecolor(COLORS['bg'])
        
        episodes = range(1, len(rewards)+1)
        ax.plot(episodes, rewards, color=color, alpha=0.15, linewidth=0.8)
        
        window = 20 if len(rewards) > 20 else 5
        smoothed = smooth_data(rewards, window=window)
        ax.plot(range(window, len(rewards)+1), smoothed, 
                color=color, linewidth=2.5, label=f'Moving Avg (w={window})')
        
        ax.axhline(y=475, color=COLORS['success'], linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='Solved (475)')
        ax.axhspan(475, 500, alpha=0.1, color=COLORS['success'])
        
        for i, r in enumerate(rewards):
            if r >= 475:
                ax.scatter(i+1, r, color=COLORS['success'], s=20, zorder=5)
        
        max_r = max(rewards)
        last100 = sum(rewards[-100:])/min(100, len(rewards))
        solved = sum(1 for r in rewards if r >= 475)
        
        ax.axhline(y=last100, color='orange', linestyle=':', linewidth=1, alpha=0.8)
        
        ax.set_xlabel('Episodes', fontsize=11, color='#2C3E50')
        ax.set_ylabel('Reward', fontsize=11, color='#2C3E50')
        ax.set_title(f'{name} Training', fontsize=13, fontweight='bold', color='#2C3E50')
        ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
        ax.set_ylim(bottom=0, top=550)
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='#2C3E50')
        
        textstr = f'Ep: {len(rewards)}\nMax: {max_r:.0f}\nSolved: {solved}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.5, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout(pad=2.0)
    plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print('[OK] results/training_curves.png (combined)')

if __name__ == '__main__':
    print('Generating training curves...\n')
    
    # 生成单独的DQN和PPO图
    with open('results/dqn_train_log.json') as f:
        dqn_rewards = json.load(f)['episode_rewards']
    with open('results/ppo_train_log.json') as f:
        ppo_rewards = json.load(f)['episode_rewards']
    
    plot_training_curve('DQN', COLORS['dqn'], dqn_rewards, 'results/dqn_training.png')
    plot_training_curve('PPO', COLORS['ppo'], ppo_rewards, 'results/ppo_training.png')
    
    # 生成组合图
    plot_combined_curves()
    
    print('\nDone!')
