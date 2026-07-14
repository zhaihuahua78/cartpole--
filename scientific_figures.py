"""专业科学图表 — 发表级RL训练曲线、对比图、箱线图"""

import json, os, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ========== 发表级 rcParams ==========
mpl.rcParams.update({
    'figure.figsize': (5.5, 3.8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'lines.linewidth': 1.2,
    'lines.markersize': 3.0,
    'axes.linewidth': 0.8,
    'axes.prop_cycle': mpl.cycler(color=['#4477AA','#EE6677','#228833','#CCBB44','#66CCEE','#AA3377','#BBBBBB']),
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'grid.linewidth': 0.4,
    'grid.alpha': 0.3,
    'grid.linestyle': '-',
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': '0.7',
    'legend.framealpha': 0.9,
    'legend.borderpad': 0.3,
})

COLORS = mpl.rcParams['axes.prop_cycle'].by_key()['color']
BG = '#FFFFFF'

def smooth(data, w=20):
    if len(data) < w: return data
    k = np.ones(w)/w
    return np.convolve(data, k, mode='valid')

# ========== 图1: DQN+PPO训练曲线 ==========
def fig_training_curves():
    with open('results/dqn_train_log.json') as f: dr = json.load(f)['episode_rewards']
    with open('results/ppo_train_log.json') as f: pr = json.load(f)['episode_rewards']

    fig, axes = plt.subplots(1,2, figsize=(10,3.8), facecolor=BG)

    for ax, (name, rewards, c) in zip(axes, [('PPO',pr,COLORS[1]),('DQN',dr,COLORS[0])]):
        steps = np.arange(1,len(rewards)+1)
        ax.plot(steps, rewards, color=c, alpha=0.12, lw=0.5)
        sw = 10 if len(rewards)<1000 else 20
        sm = smooth(rewards, sw)
        ax.plot(range(sw, len(rewards)+1), sm, color=c, lw=1.5, label=f'Smoothed (w={sw})')
        ax.fill_between(range(sw, len(rewards)+1), sm*0.85, sm*1.15, alpha=0.08, color=c)
        ax.axhline(475, color='gray', ls='--', lw=0.8, alpha=0.5, label='Solve (475)')
        ax.axhline(500, color='gray', ls=':', lw=0.5, alpha=0.3)
        # mark solved
        for i,r in enumerate(rewards):
            if r >= 475: ax.scatter(i+1, r, color=c, s=8, zorder=5, alpha=0.6)
        ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
        ax.set_title(f'{name} Training', fontweight='bold')
        ax.legend(loc='lower right'); ax.set_ylim(0,550); ax.minorticks_on()
        ax.grid(True, alpha=0.25, lw=0.3)
        # info box
        txt = f'N={len(rewards)}\nMax={max(rewards):.0f}\nSolved={sum(1 for r in rewards if r>=475)}'
        ax.text(0.97,0.45,txt,transform=ax.transAxes,fontsize=7,va='top',ha='right',
                bbox=dict(boxstyle='round',facecolor='white',alpha=0.85,edgecolor='gray',lw=0.5))

    plt.tight_layout(pad=1.5)
    fig.savefig('results/training_curves.pdf', facecolor=BG); fig.savefig('results/training_curves.png',dpi=300)
    plt.close(); print('[OK] training_curves')

# ========== 图2: 基线对比 (bar+strip) ==========
def fig_baseline_comparison():
    with open('results/baseline_comparison.json') as f: data = json.load(f)
    keys = ['random','rule_based','lqr','dqn','ppo']
    labels = ['Random','Rule','LQR','DQN','PPO']
    means = [data[k]['mean_reward'] for k in keys]
    stds  = [data[k].get('std_reward',0) for k in keys]

    fig, ax = plt.subplots(figsize=(5.5,3.8), facecolor=BG)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, 0.55, color=COLORS[:5], edgecolor='white', lw=0.8, zorder=3)
    ax.errorbar(x, means, yerr=stds, fmt='none', color='gray', capsize=4, lw=1, zorder=4)
    for bar, m in zip(bars,means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10, f'{m:.0f}', ha='center', fontsize=8, fontweight='bold')
    ax.axhline(475, color='#228833', ls='--', lw=1, alpha=0.7, label='Solve threshold (475)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Average Reward', fontsize=10)
    ax.set_title('Control Method Performance', fontweight='bold')
    ax.legend(fontsize=8); ax.set_ylim(0,580); ax.minorticks_on()
    ax.grid(axis='y', alpha=0.25, lw=0.3); ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    fig.savefig('results/baseline_comparison.pdf'); fig.savefig('results/baseline_comparison.png',dpi=300)
    plt.close(); print('[OK] baseline_comparison')

# ========== 图3: 训练前后对比 (子图: bar+box) ==========
def fig_before_after():
    import gymnasium as gym
    from stable_baselines3 import PPO, DQN

    # collect data
    env=gym.make('CartPole-v1')
    pre=[0]*20
    for i in range(20):
        o,_=env.reset(seed=i); d=False; r=0
        while not d: a=env.action_space.sample(); o,rr,t,tr,_=env.step(a); d=t or tr; r+=rr
        pre[i]=r
    env.close()

    post_d, post_p = [], []
    for algo,path,lst in [('dqn','models/dqn_cartpole',post_d),('ppo','models/ppo_cartpole',post_p)]:
        if os.path.exists(path+'.zip') or os.path.exists(path):
            m = DQN.load(path) if algo=='dqn' else PPO.load(path)
            env=gym.make('CartPole-v1')
            for i in range(20):
                o,_=env.reset(seed=i); d=False; r=0
                while not d:
                    a,_=m.predict(o,deterministic=True); o,rr,t,tr,_=env.step(int(a)); d=t or tr; r+=rr
                lst.append(r)
            env.close()

    fig, axes = plt.subplots(1,2, figsize=(8.5,3.8), facecolor=BG)
    # bar
    ax=axes[0]
    cats=['Pre\n(Random)']; vals=[np.mean(pre)]; cls=[COLORS[6]]
    if post_d: cats.append('Post\n(DQN)'); vals.append(np.mean(post_d)); cls.append(COLORS[0])
    if post_p: cats.append('Post\n(PPO)'); vals.append(np.mean(post_p)); cls.append(COLORS[1])
    x=np.arange(len(cats))
    bars=ax.bar(x,vals,0.45,color=cls,edgecolor='white',lw=0.8,zorder=3)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,b.get_height()+8,f'{v:.0f}',ha='center',fontsize=9,fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(cats,fontsize=9); ax.set_ylabel('Mean Reward')
    ax.set_title('Average Performance',fontweight='bold'); ax.set_ylim(0,580)
    ax.grid(axis='y',alpha=0.25,lw=0.3); ax.set_axisbelow(True); ax.minorticks_on()
    # box
    ax=axes[1]
    box_data=[pre]; box_lbls=['Pre']; box_cls=[COLORS[6]]
    if post_d: box_data.append(post_d); box_lbls.append('DQN'); box_cls.append(COLORS[0])
    if post_p: box_data.append(post_p); box_lbls.append('PPO'); box_cls.append(COLORS[1])
    bp=ax.boxplot(box_data,tick_labels=box_lbls,patch_artist=True,widths=0.4,
                  medianprops=dict(color='darkred',lw=1.5),
                  flierprops=dict(marker='o',markersize=3,alpha=0.4))
    for p,c in zip(bp['boxes'],box_cls): p.set_facecolor(c); p.set_alpha(0.6)
    # overlay points
    for i,d in enumerate(box_data):
        ax.scatter(np.random.normal(i+1,0.04,len(d)), d, color='black', s=8, alpha=0.3, zorder=5)
    ax.set_ylabel('Episode Reward'); ax.set_title('Reward Distribution',fontweight='bold')
    ax.grid(axis='y',alpha=0.25,lw=0.3); ax.set_axisbelow(True); ax.minorticks_on()
    plt.tight_layout(pad=1.5)
    fig.savefig('results/before_after_comparison.pdf'); fig.savefig('results/before_after_comparison.png',dpi=300)
    plt.close(); print('[OK] before_after_comparison')

# ========== 图4: 奖励函数对比 ==========
def fig_reward_comparison():
    path='results/reward_comparison.json'
    if not os.path.exists(path): print('[SKIP] reward_comparison'); return
    with open(path) as f: data = json.load(f)
    keys=['default','custom_v1','custom_v2']
    labels=[data[k]['label'] for k in keys if k in data]
    means=[data[k].get('mean_eval',0) for k in keys if k in data]
    if not labels: return

    fig, ax = plt.subplots(figsize=(5.5,3.8), facecolor=BG)
    x=np.arange(len(labels))
    bars=ax.bar(x,means,0.45,color=COLORS[0:3],edgecolor='white',lw=0.8,zorder=3)
    for b,m in zip(bars,means): ax.text(b.get_x()+b.get_width()/2,b.get_height()+5,f'{m:.0f}',ha='center',fontsize=9,fontweight='bold')
    ax.axhline(475,color='#228833',ls='--',lw=1,alpha=0.7,label='Solve (475)')
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=7,rotation=10,ha='right')
    ax.set_ylabel('Mean Eval Reward'); ax.set_title('Reward Function Comparison',fontweight='bold')
    ax.set_ylim(0,550); ax.legend(fontsize=8); ax.minorticks_on()
    ax.grid(axis='y',alpha=0.25,lw=0.3); ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    fig.savefig('results/reward_comparison.pdf'); fig.savefig('results/reward_comparison.png',dpi=300)
    plt.close(); print('[OK] reward_comparison')

# ========== 图5: DQN vs PPO 对比组合图 ==========
def fig_comparison_grid():
    with open('results/dqn_train_log.json') as f: dr=json.load(f)['episode_rewards']
    with open('results/ppo_train_log.json') as f: pr=json.load(f)['episode_rewards']
    with open('results/baseline_comparison.json') as f: bd=json.load(f)

    fig = plt.figure(figsize=(11,7.5), facecolor=BG)
    gs = fig.add_gridspec(2,3, hspace=0.35, wspace=0.35)

    # (0,0): PPO curve
    ax=fig.add_subplot(gs[0,0]); ax.set_facecolor(BG)
    steps=np.arange(1,len(pr)+1); sm=smooth(pr,10)
    ax.plot(steps,pr,color=COLORS[1],alpha=0.1,lw=0.5)
    ax.plot(range(10,len(pr)+1),sm,color=COLORS[1],lw=1.5)
    ax.fill_between(range(10,len(pr)+1),sm*0.9,sm*1.1,alpha=0.08,color=COLORS[1])
    ax.axhline(475,color='gray',ls='--',lw=0.8); ax.set_ylim(0,550)
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward'); ax.set_title('(a) PPO Training',fontweight='bold')
    ax.minorticks_on(); ax.grid(alpha=0.25,lw=0.3)

    # (0,1): DQN curve
    ax=fig.add_subplot(gs[0,1]); ax.set_facecolor(BG)
    steps=np.arange(1,len(dr)+1); sm=smooth(dr,20)
    ax.plot(steps,dr,color=COLORS[0],alpha=0.1,lw=0.5)
    ax.plot(range(20,len(dr)+1),sm,color=COLORS[0],lw=1.5)
    ax.fill_between(range(20,len(dr)+1),sm*0.85,sm*1.15,alpha=0.08,color=COLORS[0])
    for i,r in enumerate(dr):
        if r>=475: ax.scatter(i+1,r,color=COLORS[0],s=6,alpha=0.5,zorder=5)
    ax.axhline(475,color='gray',ls='--',lw=0.8); ax.set_ylim(0,550)
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward'); ax.set_title('(b) DQN Training',fontweight='bold')
    ax.minorticks_on(); ax.grid(alpha=0.25,lw=0.3)

    # (0,2): Baseline bar
    ax=fig.add_subplot(gs[0,2]); ax.set_facecolor(BG)
    keys=['random','rule_based','lqr','dqn','ppo']; lbls=['Random','Rule','LQR','DQN','PPO']
    means=[bd[k]['mean_reward'] for k in keys]; stds=[bd[k].get('std_reward',0) for k in keys]
    x=np.arange(len(lbls))
    bars=ax.bar(x,means,0.55,color=COLORS[:5],edgecolor='white',lw=0.8,zorder=3)
    ax.errorbar(x,means,yerr=stds,fmt='none',color='gray',capsize=3,lw=1,zorder=4)
    for b,m in zip(bars,means): ax.text(b.get_x()+b.get_width()/2,b.get_height()+8,f'{m:.0f}',ha='center',fontsize=7,fontweight='bold')
    ax.axhline(475,color='#228833',ls='--',lw=1,alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(lbls,fontsize=8); ax.set_ylabel('Mean Reward')
    ax.set_title('(c) Method Comparison',fontweight='bold'); ax.set_ylim(0,580)
    ax.minorticks_on(); ax.grid(axis='y',alpha=0.25,lw=0.3); ax.set_axisbelow(True)

    # (1,0): convergence speed
    ax=fig.add_subplot(gs[1,0]); ax.set_facecolor(BG)
    # compute when each algo first reached 475
    ppo_conv = next((i for i,r in enumerate(pr) if r>=475), len(pr))
    dqn_conv = next((i for i,r in enumerate(dr) if r>=475), len(dr))
    ax.bar(['PPO','DQN'],[ppo_conv,dqn_conv],0.4,color=[COLORS[1],COLORS[0]],edgecolor='white',lw=0.8,zorder=3)
    ax.text(0,ppo_conv+20,f'{ppo_conv}',ha='center',fontsize=9,fontweight='bold')
    ax.text(1,dqn_conv+20,f'{dqn_conv}',ha='center',fontsize=9,fontweight='bold')
    ax.set_ylabel('Episode to Solve'); ax.set_title('(d) Convergence Speed',fontweight='bold')
    ax.minorticks_on(); ax.grid(axis='y',alpha=0.25,lw=0.3); ax.set_axisbelow(True)

    # (1,1): stability (CV of last 100)
    ax=fig.add_subplot(gs[1,1]); ax.set_facecolor(BG)
    dqn_last100=dr[-100:]
    ppo_last100=pr[-100:]
    dqn_cv=np.std(dqn_last100)/(np.mean(dqn_last100)+1e-8)
    ppo_cv=np.std(ppo_last100)/(np.mean(ppo_last100)+1e-8)
    ax.bar(['PPO','DQN'],[ppo_cv,dqn_cv],0.4,color=[COLORS[1],COLORS[0]],edgecolor='white',lw=0.8,zorder=3)
    ax.text(0,ppo_cv+0.01,f'{ppo_cv:.2f}',ha='center',fontsize=9,fontweight='bold')
    ax.text(1,dqn_cv+0.01,f'{dqn_cv:.2f}',ha='center',fontsize=9,fontweight='bold')
    ax.set_ylabel('CV (lower=more stable)'); ax.set_title('(e) Training Stability',fontweight='bold')
    ax.minorticks_on(); ax.grid(axis='y',alpha=0.25,lw=0.3); ax.set_axisbelow(True)

    # (1,2): sample efficiency
    ax=fig.add_subplot(gs[1,2]); ax.set_facecolor(BG)
    dqn_steps=150000; ppo_steps=50000
    dqn_eff = np.mean(dr[-100:])/dqn_steps*1000
    ppo_eff = np.mean(pr[-100:])/ppo_steps*1000
    ax.bar(['PPO','DQN'],[ppo_eff,dqn_eff],0.4,color=[COLORS[1],COLORS[0]],edgecolor='white',lw=0.8,zorder=3)
    ax.text(0,ppo_eff+0.2,f'{ppo_eff:.1f}e-3',ha='center',fontsize=8,fontweight='bold')
    ax.text(1,dqn_eff+0.2,f'{dqn_eff:.1f}e-3',ha='center',fontsize=8,fontweight='bold')
    ax.set_ylabel('Reward/1k Steps'); ax.set_title('(f) Sample Efficiency',fontweight='bold')
    ax.minorticks_on(); ax.grid(axis='y',alpha=0.25,lw=0.3); ax.set_axisbelow(True)

    plt.tight_layout(pad=1.0)
    fig.savefig('results/comparison_grid.pdf'); fig.savefig('results/comparison_grid.png',dpi=300)
    plt.close(); print('[OK] comparison_grid')

if __name__ == '__main__':
    print('Generating professional scientific figures...\n')
    fig_training_curves()
    fig_baseline_comparison()
    fig_before_after()
    fig_reward_comparison()
    fig_comparison_grid()
    print('\nAll figures generated!')
