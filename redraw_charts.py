#!/usr/bin/env python
"""
重新绘制 CartPole 项目的可视化图表
生成专业美观的训练曲线、基线对比和训练前后对比图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置全局样式
plt.style.use('seaborn-v0_8-whitegrid')

# 颜色主题
COLORS = {
    'dqn': '#2E86AB',      # 蓝色
    'ppo': '#A23B72',      # 紫红色
    'lqr': '#F18F01',      # 橙色
    'rule': '#C73E1D',     # 红色
    'random': '#3B1F2B',   # 深灰
    'bg': '#FAFAFA',       # 背景色
    'grid': '#E8E8E8',     # 网格色
    'text': '#2C3E50',     # 文字色
    'accent1': '#27AE60',  # 绿色（成功）
    'accent2': '#E74C3C',  # 红色（失败）
}

def smooth_data(data, window=10):
    """移动平均平滑"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_training_curves():
    """绘制 DQN 与 PPO 训练奖励曲线对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS['bg'])
    
    # 读取数据
    with open('results/dqn_train_log.json') as f:
        dqn_data = json.load(f)
    with open('results/ppo_train_log.json') as f:
        ppo_data = json.load(f)
    
    # DQN 训练曲线
    ax1 = axes[0]
    ax1.set_facecolor(COLORS['bg'])
    dqn_rewards = dqn_data['episode_rewards']
    
    # 绘制原始数据（半透明）
    ax1.plot(range(1, len(dqn_rewards)+1), dqn_rewards, 
             color=COLORS['dqn'], alpha=0.15, linewidth=0.8)
    
    # 绘制平滑曲线
    if len(dqn_rewards) > 20:
        smoothed = smooth_data(dqn_rewards, window=20)
        ax1.plot(range(20, len(dqn_rewards)+1), smoothed, 
                 color=COLORS['dqn'], linewidth=2.5, label='移动平均 (w=20)')
    
    # 添加阈值线
    ax1.axhline(y=475, color=COLORS['accent1'], linestyle='--', 
                linewidth=1.5, alpha=0.7, label='解决阈值 (475)')
    ax1.axhline(y=500, color=COLORS['accent1'], linestyle='-', 
                linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('训练回合数', fontsize=11, color=COLORS['text'])
    ax1.set_ylabel('奖励', fontsize=11, color=COLORS['text'])
    ax1.set_title('DQN 训练曲线', fontsize=13, fontweight='bold', 
                  color=COLORS['text'], pad=10)
    ax1.legend(loc='upper left', framealpha=0.9, edgecolor='none')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'])
    ax1.tick_params(colors=COLORS['text'])
    for spine in ax1.spines.values():
        spine.set_color(COLORS['grid'])
    
    # PPO 训练曲线
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['bg'])
    ppo_rewards = ppo_data['episode_rewards']
    
    ax2.plot(range(1, len(ppo_rewards)+1), ppo_rewards, 
             color=COLORS['ppo'], alpha=0.15, linewidth=0.8)
    
    if len(ppo_rewards) > 10:
        smoothed_ppo = smooth_data(ppo_rewards, window=10)
        ax2.plot(range(10, len(ppo_rewards)+1), smoothed_ppo, 
                 color=COLORS['ppo'], linewidth=2.5, label='移动平均 (w=10)')
    
    ax2.axhline(y=475, color=COLORS['accent1'], linestyle='--', 
                linewidth=1.5, alpha=0.7, label='解决阈值 (475)')
    ax2.axhline(y=500, color=COLORS['accent1'], linestyle='-', 
                linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('训练回合数', fontsize=11, color=COLORS['text'])
    ax2.set_ylabel('奖励', fontsize=11, color=COLORS['text'])
    ax2.set_title('PPO 训练曲线', fontsize=13, fontweight='bold', 
                  color=COLORS['text'], pad=10)
    ax2.legend(loc='upper left', framealpha=0.9, edgecolor='none')
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'])
    ax2.tick_params(colors=COLORS['text'])
    for spine in ax2.spines.values():
        spine.set_color(COLORS['grid'])
    
    plt.tight_layout(pad=2.0)
    plt.savefig('results/training_curves.png', dpi=150, 
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print('[OK] training_curves.png 已生成')


def plot_baseline_comparison():
    """绘制控制方法对比图"""
    # 读取基线对比数据
    with open('results/baseline_comparison.json') as f:
        data = json.load(f)
    
    # 提取数据
    methods = []
    rewards = []
    colors = []
    error_bars = []
    
    method_map = {
        'random': ('随机策略', COLORS['random']),
        'rule_based': ('规则控制器', COLORS['rule']),
        'lqr': ('LQR 控制器', COLORS['lqr']),
        'dqn': ('DQN', COLORS['dqn']),
        'ppo': ('PPO', COLORS['ppo']),
    }
    
    for key in ['random', 'rule_based', 'lqr', 'dqn', 'ppo']:
        if key in data:
            methods.append(method_map[key][0])
            rewards.append(data[key]['mean_reward'])
            colors.append(method_map[key][1])
            error_bars.append(data[key].get('std_reward', 0))
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    x = np.arange(len(methods))
    width = 0.6
    
    # 绘制柱状图
    bars = ax.bar(x, rewards, width, color=colors, edgecolor='white', 
                  linewidth=1.5, zorder=3)
    
    # 添加误差棒
    if any(e > 0 for e in error_bars):
        ax.errorbar(x, rewards, yerr=error_bars, fmt='none', 
                    color=COLORS['text'], capsize=5, capthick=1.5, linewidth=1.5)
    
    # 在柱子上方添加数值
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 8,
                f'{reward:.0f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color=COLORS['text'])
    
    # 添加解决阈值线
    ax.axhline(y=475, color=COLORS['accent1'], linestyle='--', 
               linewidth=1.5, alpha=0.7, label='解决阈值 (475)')
    
    ax.set_xlabel('控制方法', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('平均奖励', fontsize=12, color=COLORS['text'])
    ax.set_title('控制方法性能对比', fontsize=14, fontweight='bold', 
                 color=COLORS['text'], pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, color=COLORS['text'])
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')
    ax.set_ylim(bottom=0, top=max(rewards) * 1.15)
    ax.grid(True, axis='y', alpha=0.3, color=COLORS['grid'], zorder=0)
    ax.tick_params(colors=COLORS['text'])
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    
    plt.tight_layout()
    plt.savefig('results/baseline_comparison.png', dpi=150, 
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print('[OK] baseline_comparison.png 已生成')


def plot_before_after():
    """绘制训练前后控制效果对比"""
    import gymnasium as gym
    
    # 训练前：随机策略
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
    
    # 训练后：加载模型
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
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=COLORS['bg'])
    
    # 左图：柱状图对比
    ax1 = axes[0]
    ax1.set_facecolor(COLORS['bg'])
    
    categories = ['训练前\n(随机策略)']
    values = [np.mean(pre_rewards)]
    bar_colors = [COLORS['random']]
    
    if post_rewards_dqn:
        categories.append('训练后\n(DQN)')
        values.append(np.mean(post_rewards_dqn))
        bar_colors.append(COLORS['dqn'])
    if post_rewards_ppo:
        categories.append('训练后\n(PPO)')
        values.append(np.mean(post_rewards_ppo))
        bar_colors.append(COLORS['ppo'])
    
    x = np.arange(len(categories))
    bars = ax1.bar(x, values, 0.5, color=bar_colors, edgecolor='white', 
                   linewidth=1.5, zorder=3)
    
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 8,
                f'{val:.0f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color=COLORS['text'])
    
    ax1.set_ylabel('平均奖励', fontsize=11, color=COLORS['text'])
    ax1.set_title('训练前后控制效果对比', fontsize=13, fontweight='bold', 
                  color=COLORS['text'], pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10, color=COLORS['text'])
    ax1.grid(True, axis='y', alpha=0.3, color=COLORS['grid'], zorder=0)
    ax1.set_ylim(bottom=0)
    ax1.tick_params(colors=COLORS['text'])
    for spine in ax1.spines.values():
        spine.set_color(COLORS['grid'])
    
    # 右图：箱线图
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['bg'])
    
    box_data = [pre_rewards]
    box_labels = ['训练前']
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
    
    ax2.set_ylabel('回合奖励', fontsize=11, color=COLORS['text'])
    ax2.set_title('奖励分布', fontsize=13, fontweight='bold', 
                  color=COLORS['text'], pad=10)
    ax2.grid(True, axis='y', alpha=0.3, color=COLORS['grid'])
    ax2.tick_params(colors=COLORS['text'])
    for spine in ax2.spines.values():
        spine.set_color(COLORS['grid'])
    
    plt.tight_layout(pad=2.0)
    plt.savefig('results/before_after_comparison.png', dpi=150, 
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print('[OK] before_after_comparison.png 已生成')


def plot_reward_comparison():
    """绘制奖励函数对比图"""
    # 检查是否有奖励对比数据
    reward_file = 'results/reward_comparison.json'
    if not os.path.exists(reward_file):
        print('[WARN] reward_comparison.json 不存在，跳过')
        return
    
    with open(reward_file) as f:
        data = json.load(f)
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    methods = []
    rewards = []
    colors_list = []
    
    color_map = {
        'default': '#2E86AB',
        'custom_v1': '#A23B72',
        'custom_v2': '#F18F01',
    }
    
    for key in ['default', 'custom_v1', 'custom_v2']:
        if key in data:
            methods.append(data[key]['label'])
            rewards.append(data[key]['mean_eval'])
            colors_list.append(color_map.get(key, '#666666'))
    
    if not methods:
        return
    
    x = np.arange(len(methods))
    bars = ax.bar(x, rewards, 0.5, color=colors_list, edgecolor='white', 
                  linewidth=1.5, zorder=3)
    
    for bar, reward in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{reward:.0f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color=COLORS['text'])
    
    ax.axhline(y=475, color=COLORS['accent1'], linestyle='--', 
               linewidth=1.5, alpha=0.7, label='解决阈值 (475)')
    
    ax.set_xlabel('奖励函数', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('评估均值', fontsize=12, color=COLORS['text'])
    ax.set_title('奖励函数设计对比', fontsize=14, fontweight='bold', 
                 color=COLORS['text'], pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, color=COLORS['text'], rotation=15, ha='right')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')
    ax.set_ylim(bottom=0, top=550)
    ax.grid(True, axis='y', alpha=0.3, color=COLORS['grid'], zorder=0)
    ax.tick_params(colors=COLORS['text'])
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    
    plt.tight_layout()
    plt.savefig('results/reward_comparison.png', dpi=150, 
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print('[OK] reward_comparison.png 已生成')


if __name__ == '__main__':
    print('[*] 开始重新绘制图表...\n')
    
    plot_training_curves()
    plot_baseline_comparison()
    plot_before_after()
    plot_reward_comparison()
    
    print('\n[OK] 所有图表绘制完成！')
