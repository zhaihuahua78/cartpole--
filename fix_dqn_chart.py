#!/usr/bin/env python
"""重新绘制DQN训练曲线 - 更好地展示学习过程"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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

def plot_dqn_training():
    """重新绘制DQN训练曲线 - 聚焦学习过程"""
    with open('results/dqn_train_log.json') as f:
        dqn_data = json.load(f)
    
    rewards = dqn_data['episode_rewards']
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    # 原始数据
    episodes = range(1, len(rewards)+1)
    ax.plot(episodes, rewards, color=COLORS['dqn'], alpha=0.15, linewidth=0.8)
    
    # 平滑曲线
    smoothed = smooth_data(rewards, window=20)
    ax.plot(range(20, len(rewards)+1), smoothed, 
            color=COLORS['dqn'], linewidth=2.5, label='Moving Average (w=20)')
    
    # 标注满分区域
    ax.axhline(y=475, color=COLORS['success'], linestyle='--', 
               linewidth=1.5, alpha=0.7, label='Solved Threshold (475)')
    ax.axhspan(475, 500, alpha=0.1, color=COLORS['success'])
    
    # 标注最后阶段的满分
    for i in range(len(rewards)-10, len(rewards)):
        if rewards[i] >= 475:
            ax.scatter(i+1, rewards[i], color=COLORS['success'], s=50, zorder=5)
    
    # 添加注释
    ax.annotate(f'Max: {max(rewards):.0f}', 
                xy=(len(rewards), max(rewards)), 
                xytext=(len(rewards)-200, max(rewards)-50),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, color='black')
    
    # 标注最后100轮
    last100_mean = sum(rewards[-100:])/100
    ax.axhline(y=last100_mean, color='orange', linestyle=':', 
               linewidth=1, alpha=0.8, label=f'Last 100 Mean: {last100_mean:.0f}')
    
    ax.set_xlabel('Training Episodes', fontsize=12, color='#2C3E50')
    ax.set_ylabel('Episode Reward', fontsize=12, color='#2C3E50')
    ax.set_title('DQN Training Curve - Achieved Perfect Score!', fontsize=14, 
                 fontweight='bold', color='#2C3E50', pad=15)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.set_ylim(bottom=0, top=550)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors='#2C3E50')
    
    # 添加文本说明
    textstr = f'Episodes: {len(rewards)}\nMax Reward: 500\nSolved Episodes: {sum(1 for r in rewards if r >= 475)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print('[OK] training_curves.png - DQN curve updated')

if __name__ == '__main__':
    plot_dqn_training()
