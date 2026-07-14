"""
Nature-style scientific figures for CartPole RL report.
Strictly follows Nature期刊 figure conventions:
- Black/white/gray base + single accent color
- No chartjunk, no redundant decorations
- Axis labels capitalized, no legend border
- Each figure standalone with its own context
"""
import json, os, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============ Nature journal style rcParams ============
NATURE_GRAY = '#4D4D4D'       # dark gray for text/axes
ACCENT_BLUE = '#2166AC'       # single accent color (Nature blue)
ACCENT_RED  = '#B2182B'       # secondary accent
LIGHT_GRAY = '#E0E0E0'        # grid lines
WHITE = '#FFFFFF'

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'axes.edgecolor': NATURE_GRAY,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.color': NATURE_GRAY,
    'ytick.color': NATURE_GRAY,
    'text.color': NATURE_GRAY,
    'axes.labelcolor': NATURE_GRAY,
    'legend.edgecolor': 'none',
    'legend.frameon': False,
    'legend.borderpad': 0.2,
})

def smooth(data, w=20):
    if len(data) < w: return data
    return np.convolve(data, np.ones(w)/w, mode='valid')

def nature_clean(ax):
    """Strip the axes down to Nature style: only left+bottom spines, thin frames."""
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['left'].set_color(NATURE_GRAY)
    ax.spines['bottom'].set_color(NATURE_GRAY)
    ax.tick_params(colors=NATURE_GRAY)

# ========== FIGURE 1: PPO Training Curve (Section 2.2) ==========
def fig1_ppo_training():
    with open('results/ppo_train_log.json') as f:
        rewards = json.load(f)['episode_rewards']
    
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    nature_clean(ax)
    
    episodes = np.arange(1, len(rewards) + 1)
    
    # raw data as faint gray dots
    step = max(1, len(rewards)//400)
    ax.scatter(episodes[::step], rewards[::step], color='#B0B0B0', s=2, alpha=0.5, rasterized=True)
    
    # smoothed line in accent blue
    sm = smooth(rewards, 10)
    ax.plot(range(10, len(rewards)+1), sm, color=ACCENT_BLUE, linewidth=1.2)
    
    # solve threshold
    ax.axhline(y=475, color=NATURE_GRAY, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.text(len(rewards)*0.98, 480, 'Solve threshold', fontsize=6, color=NATURE_GRAY, ha='right', va='bottom')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_ylim(0, 550)
    ax.set_xlim(0, len(rewards))
    
    # annotation
    ax.annotate(f'First solved: episode {next(i for i,r in enumerate(rewards) if r>=475)}',
                xy=(next(i for i,r in enumerate(rewards) if r>=475), 475),
                xytext=(next(i for i,r in enumerate(rewards) if r>=475)+80, 380),
                arrowprops=dict(arrowstyle='->', color=NATURE_GRAY, lw=0.6),
                fontsize=6, color=NATURE_GRAY)
    
    plt.tight_layout(pad=0.3)
    for fmt in ['png','pdf']:
        fig.savefig(f'results/fig1_ppo_training.{fmt}', dpi=300, facecolor=WHITE)
    plt.close()
    print('[OK] fig1_ppo_training')

# ========== FIGURE 2: DQN Training Curve (Section 2.2) ==========
def fig2_dqn_training():
    with open('results/dqn_train_log.json') as f:
        rewards = json.load(f)['episode_rewards']
    
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    nature_clean(ax)
    
    episodes = np.arange(1, len(rewards) + 1)
    step = max(1, len(rewards)//400)
    ax.scatter(episodes[::step], rewards[::step], color='#B0B0B0', s=2, alpha=0.5, rasterized=True)
    
    sm = smooth(rewards, 20)
    ax.plot(range(20, len(rewards)+1), sm, color=ACCENT_RED, linewidth=1.2)
    
    # mark solved episodes
    solved_eps = [i+1 for i,r in enumerate(rewards) if r >= 475]
    for ep in solved_eps:
        ax.axvline(x=ep, color=ACCENT_RED, linewidth=0.4, alpha=0.3)
    
    ax.axhline(y=475, color=NATURE_GRAY, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.text(len(rewards)*0.98, 480, 'Solve threshold', fontsize=6, color=NATURE_GRAY, ha='right', va='bottom')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_ylim(0, 550)
    ax.set_xlim(0, len(rewards))
    
    # info inset
    txt = f'Episodes: {len(rewards)}\nMax reward: {max(rewards):.0f}\nSolved: {len(solved_eps)} times\nLast 100 mean: {np.mean(rewards[-100:]):.0f}'
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=6, color=NATURE_GRAY,
            va='top', family='monospace')
    
    plt.tight_layout(pad=0.3)
    for fmt in ['png','pdf']:
        fig.savefig(f'results/fig2_dqn_training.{fmt}', dpi=300, facecolor=WHITE)
    plt.close()
    print('[OK] fig2_dqn_training')

# ========== FIGURE 3: Method Comparison (Section 3.3) ==========
def fig3_method_comparison():
    with open('results/baseline_comparison.json') as f:
        data = json.load(f)
    
    keys = ['random', 'lqr', 'rule_based', 'dqn', 'ppo']
    labels = ['Random', 'LQR', 'Rule-based', 'DQN', 'PPO']
    means = [data[k]['mean_reward'] for k in keys]
    stds  = [data[k].get('std_reward', 0) for k in keys]
    
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    nature_clean(ax)
    
    x = np.arange(len(labels))
    # All gray except PPO in blue
    bar_colors = ['#9E9E9E']*4 + [ACCENT_BLUE]
    
    bars = ax.bar(x, means, 0.55, color=bar_colors, edgecolor=WHITE, linewidth=0.5, zorder=3)
    ax.errorbar(x, means, yerr=stds, fmt='none', color=NATURE_GRAY, capsize=3, lw=0.8, zorder=4)
    
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f'{m:.0f}', ha='center', fontsize=7, color=NATURE_GRAY)
    
    ax.axhline(y=475, color=NATURE_GRAY, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.text(4.5, 480, 'Solve', fontsize=6, color=NATURE_GRAY, ha='right')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Mean reward')
    ax.set_ylim(0, 580)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    
    plt.tight_layout(pad=0.3)
    for fmt in ['png','pdf']:
        fig.savefig(f'results/fig3_method_comparison.{fmt}', dpi=300, facecolor=WHITE)
    plt.close()
    print('[OK] fig3_method_comparison')

# ========== FIGURE 4: Before/After Training (Section 3.2) ==========
def fig4_before_after():
    import gymnasium as gym
    from stable_baselines3 import PPO, DQN
    
    env = gym.make('CartPole-v1')
    pre = []
    for i in range(20):
        o, _ = env.reset(seed=i); d = False; r = 0
        while not d:
            a = env.action_space.sample(); o, rr, t, tr, _ = env.step(a); d = t or tr; r += rr
        pre.append(r)
    env.close()
    
    post = []
    for algo, path in [('dqn','models/dqn_cartpole'), ('ppo','models/ppo_cartpole')]:
        if os.path.exists(path+'.zip') or os.path.exists(path):
            m = DQN.load(path) if algo=='dqn' else PPO.load(path)
            env = gym.make('CartPole-v1')
            lst = []
            for i in range(20):
                o, _ = env.reset(seed=i); d = False; r = 0
                while not d:
                    a, _ = m.predict(o, deterministic=True); o, rr, t, tr, _ = env.step(int(a)); d = t or tr; r += rr
                lst.append(r)
            post.append((algo.upper(), lst))
            env.close()
    
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    nature_clean(ax)
    
    # box plot
    all_data = [pre] + [p[1] for p in post]
    all_labels = ['Random'] + [p[0] for p in post]
    
    # Manual box plots with nature styling
    bp = ax.boxplot(all_data, patch_artist=True, widths=0.35, tick_labels=all_labels,
                    medianprops=dict(color='black', linewidth=0.8),
                    whiskerprops=dict(color=NATURE_GRAY, linewidth=0.6),
                    capprops=dict(color=NATURE_GRAY, linewidth=0.6),
                    boxprops=dict(edgecolor=NATURE_GRAY, linewidth=0.6),
                    flierprops=dict(marker='o', markersize=2, alpha=0.3, color=NATURE_GRAY))
    
    box_colors = ['#D0D0D0', '#D0D0D0', ACCENT_BLUE]
    for patch, c in zip(bp['boxes'], box_colors[:len(all_data)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    
    # overlay individual data points
    for i, d in enumerate(all_data):
        jitter = np.random.normal(0, 0.03, len(d))
        ax.scatter(np.ones(len(d))*(i+1) + jitter, d, s=6, color=NATURE_GRAY, alpha=0.4, zorder=5)
    
    ax.axhline(y=475, color=ACCENT_BLUE, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_ylabel('Episode reward')
    ax.set_ylim(0, 550)
    
    # add mean labels
    for i, d in enumerate(all_data):
        ax.text(i+1, np.mean(d)+25, f'{np.mean(d):.0f}', ha='center', fontsize=7, color=NATURE_GRAY, fontweight='bold')
    
    plt.tight_layout(pad=0.3)
    for fmt in ['png','pdf']:
        fig.savefig(f'results/fig4_before_after.{fmt}', dpi=300, facecolor=WHITE)
    plt.close()
    print('[OK] fig4_before_after')

# ========== FIGURE 5: Convergence Speed (Section 3.3) ==========
def fig5_convergence_speed():
    with open('results/dqn_train_log.json') as f: dr = json.load(f)['episode_rewards']
    with open('results/ppo_train_log.json') as f: pr = json.load(f)['episode_rewards']
    
    # Steps to first solve
    dqn_conv = next((i for i,r in enumerate(dr) if r>=475), len(dr))
    ppo_conv = next((i for i,r in enumerate(pr) if r>=475), len(pr))
    
    # Steps to final stable solve (last episode below 475)
    dqn_final = len(dr) - next((i for i,r in enumerate(reversed(dr)) if r<475), 0)
    ppo_final = len(pr) - next((i for i,r in enumerate(reversed(pr)) if r<475), 0)
    
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    nature_clean(ax)
    
    x = np.arange(2)
    width = 0.28
    
    bars1 = ax.bar(x - width/2, [ppo_conv, dqn_conv], width, color='#9E9E9E',
                   edgecolor=WHITE, linewidth=0.5, label='First solved')
    bars2 = ax.bar(x + width/2, [ppo_final, dqn_final], width, color=ACCENT_BLUE,
                   edgecolor=WHITE, linewidth=0.5, alpha=0.7, label='Stable start')
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+30,
                f'{bar.get_height():.0f}', ha='center', fontsize=7, color=NATURE_GRAY)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+30,
                f'{bar.get_height():.0f}', ha='center', fontsize=7, color=ACCENT_BLUE)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['PPO', 'DQN'])
    ax.set_ylabel('Episode')
    ax.legend(loc='upper left', fontsize=7)
    
    plt.tight_layout(pad=0.3)
    for fmt in ['png','pdf']:
        fig.savefig(f'results/fig5_convergence_speed.{fmt}', dpi=300, facecolor=WHITE)
    plt.close()
    print('[OK] fig5_convergence_speed')

# ========== FIGURE 6: Training Stability (Section 3.3) ==========
def fig6_stability():
    with open('results/dqn_train_log.json') as f: dr = json.load(f)['episode_rewards']
    with open('results/ppo_train_log.json') as f: pr = json.load(f)['episode_rewards']
    
    # Compute rolling mean and std for last 500 episodes
    def rolling_stats(data, window=100):
        means = [np.mean(data[max(0,i-window):i]) for i in range(window, len(data)+1)]
        stds  = [np.std(data[max(0,i-window):i]) for i in range(window, len(data)+1)]
        return means, stds
    
    ppo_means, ppo_stds = rolling_stats(pr)
    dqn_means, dqn_stds = rolling_stats(dr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8), facecolor=WHITE)
    
    # PPO stability
    nature_clean(ax1); ax1.set_facecolor(WHITE)
    x1 = range(100, len(pr)+1)
    ax1.plot(x1, ppo_means, color=ACCENT_BLUE, linewidth=1.0)
    ax1.fill_between(x1, np.array(ppo_means)-np.array(ppo_stds),
                     np.array(ppo_means)+np.array(ppo_stds), color=ACCENT_BLUE, alpha=0.12, linewidth=0)
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Rolling mean reward')
    ax1.set_title('PPO', fontsize=9, fontweight='normal', color=NATURE_GRAY, pad=5)
    ax1.set_ylim(0, 550)
    ax1.axhline(475, color=NATURE_GRAY, ls='--', lw=0.5, alpha=0.5)
    
    # DQN stability
    nature_clean(ax2); ax2.set_facecolor(WHITE)
    x2 = range(100, len(dr)+1)
    ax2.plot(x2, dqn_means, color=ACCENT_RED, linewidth=1.0)
    ax2.fill_between(x2, np.array(dqn_means)-np.array(dqn_stds),
                     np.array(dqn_means)+np.array(dqn_stds), color=ACCENT_RED, alpha=0.12, linewidth=0)
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Rolling mean reward')
    ax2.set_title('DQN', fontsize=9, fontweight='normal', color=NATURE_GRAY, pad=5)
    ax2.set_ylim(0, 550)
    ax2.axhline(475, color=NATURE_GRAY, ls='--', lw=0.5, alpha=0.5)
    
    plt.tight_layout(pad=0.5, w_pad=1.0)
    for fmt in ['png','pdf']:
        fig.savefig(f'results/fig6_stability.{fmt}', dpi=300, facecolor=WHITE)
    plt.close()
    print('[OK] fig6_stability')

# ========== FIGURE 7: Sample Efficiency (Section 3.3) ==========
def fig7_sample_efficiency():
    with open('results/ppo_train_log.json') as f: pr = json.load(f)['episode_rewards']
    with open('results/dqn_train_log.json') as f: dr = json.load(f)['episode_rewards']
    
    # Cumulative max reward vs. environment steps
    ppo_steps_total = 50000
    dqn_steps_total = 150000
    
    # Map episode -> environment step (approximate: total_steps / n_episodes linear)
    ppo_steps_per_ep = ppo_steps_total / len(pr)
    dqn_steps_per_ep = dqn_steps_total / len(dr)
    
    ppo_cummax = np.maximum.accumulate(pr)
    dqn_cummax = np.maximum.accumulate(dr)
    
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    nature_clean(ax)
    
    ax.plot(np.arange(1, len(pr)+1) * ppo_steps_per_ep / 1000, ppo_cummax,
            color=ACCENT_BLUE, linewidth=1.2, label='PPO (50k steps)')
    ax.plot(np.arange(1, len(dr)+1) * dqn_steps_per_ep / 1000, dqn_cummax,
            color=ACCENT_RED, linewidth=1.2, label=f'DQN ({dqn_steps_total//1000}k steps)')
    
    ax.axhline(475, color=NATURE_GRAY, ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel('Environment steps (thousands)')
    ax.set_ylabel('Best reward so far')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 550)
    
    plt.tight_layout(pad=0.3)
    for fmt in ['png','pdf']:
        fig.savefig(f'results/fig7_sample_efficiency.{fmt}', dpi=300, facecolor=WHITE)
    plt.close()
    print('[OK] fig7_sample_efficiency')

# ========== FIGURE 8: Reward Function Comparison (Section 1.2.2) ==========
def fig8_reward_comparison():
    path = 'results/reward_comparison.json'
    if not os.path.exists(path): 
        print('[SKIP] fig8_reward_comparison (no data)')
        return
    
    with open(path) as f: data = json.load(f)
    keys = ['default', 'custom_v1', 'custom_v2']
    labels = [data[k]['label'] for k in keys if k in data]
    means  = [data[k].get('mean_eval', 0) for k in keys if k in data]
    
    if not means: return
    
    # Extract training curves if available
    curves = {}
    for k in keys:
        if k in data and 'train_curve' in data[k]:
            curves[k] = data[k]['train_curve']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8), facecolor=WHITE)
    
    # Bar chart
    nature_clean(ax1); ax1.set_facecolor(WHITE)
    x = np.arange(len(labels))
    colors_bar = ['#9E9E9E', '#B0B0B0', ACCENT_BLUE]
    bars = ax1.bar(x, means, 0.45, color=colors_bar[:len(labels)], edgecolor=WHITE, linewidth=0.5)
    for bar, m in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()+10,
                f'{m:.0f}', ha='center', fontsize=7, color=NATURE_GRAY)
    ax1.axhline(475, color=NATURE_GRAY, ls='--', lw=0.5, alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6, rotation=15, ha='right')
    ax1.set_ylabel('Mean evaluation reward')
    ax1.set_ylim(0, 550)
    
    # Training curves
    if curves:
        nature_clean(ax2); ax2.set_facecolor(WHITE)
        for i, (k, label) in enumerate(zip(keys, labels)):
            if k in curves:
                r = curves[k]
                sm = smooth(r, 10) if len(r) > 10 else r
                ax2.plot(range(len(sm)), sm, linewidth=0.8, label=label,
                        color=[ACCENT_BLUE, ACCENT_RED, '#9E9E9E'][i])
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Smoothed reward')
        ax2.legend(fontsize=6)
        ax2.set_ylim(0, 550)
    
    plt.tight_layout(pad=0.5, w_pad=1.0)
    for fmt in ['png','pdf']:
        fig.savefig(f'results/fig8_reward_comparison.{fmt}', dpi=300, facecolor=WHITE)
    plt.close()
    print('[OK] fig8_reward_comparison')

if __name__ == '__main__':
    print('Generating Nature-style figures...\n')
    fig1_ppo_training()
    fig2_dqn_training()
    fig3_method_comparison()
    fig4_before_after()
    fig5_convergence_speed()
    fig6_stability()
    fig7_sample_efficiency()
    fig8_reward_comparison()
    print('\nAll Nature-style figures generated!')
