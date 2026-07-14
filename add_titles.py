"""Add descriptive titles (no Figure N numbering) back to all figures"""
import os
os.chdir("D:/大二下/AI赋能自动控制原理/AI赋能自动控制原理/AI赋能自动控制原理/task1_cartpole")

with open("nature_figures.py", "r", encoding="utf-8") as f:
    t = f.read()

# Add titles after specific ax.set_ylabel lines
replacements = [
    # Fig1: PPO training curve
    ('    ax.set_ylabel("Reward", fontsize=8)\n    ax.set_xlim(0, len(rewards)+5); ax.set_ylim(-50, 550)',
     '    ax.set_ylabel("Reward", fontsize=8)\n    ax.set_title("PPO training curve", fontsize=9, fontweight="bold", pad=8)\n    ax.set_xlim(0, len(rewards)+5); ax.set_ylim(-50, 550)'),
    
    # Fig2: A2C training curve
    ('    ax.set_ylabel("Reward", fontsize=8)\n    ax.set_xlim(-50, len(rewards)+50); ax.set_ylim(-50, 550)',
     '    ax.set_ylabel("Reward", fontsize=8)\n    ax.set_title("A2C training curve", fontsize=9, fontweight="bold", pad=8)\n    ax.set_xlim(-50, len(rewards)+50); ax.set_ylim(-50, 550)'),
    
    # Fig3: Control method comparison
    ('    ax.set_ylabel("Mean reward", fontsize=8)\n    ax.set_ylim(0, 580)',
     '    ax.set_ylabel("Mean reward", fontsize=8)\n    ax.set_title("Control method comparison", fontsize=9, fontweight="bold", pad=8)\n    ax.set_ylim(0, 580)'),
    
    # Fig4: PPO subplot title
    ('    ax1.set_xlabel("Episode", fontsize=7.5); ax1.set_ylabel("Reward", fontsize=7.5)\n\n    # A2C',
     '    ax1.set_xlabel("Episode", fontsize=7.5); ax1.set_ylabel("Reward", fontsize=7.5)\n    ax1.set_title("PPO", fontsize=8, fontweight="bold", pad=5)\n\n    # A2C'),
    
    # Fig4: A2C subplot title
    ('    ax2.set_xlabel("Episode", fontsize=7.5)\n\n    _save(fig, "fig4_training_comparison")',
     '    ax2.set_xlabel("Episode", fontsize=7.5)\n    ax2.set_title("A2C", fontsize=8, fontweight="bold", pad=5)\n\n    _save(fig, "fig4_training_comparison")'),
    
    # Fig5: Before vs after
    ('    ax.set_ylabel("Reward", fontsize=8)\n    _save(fig, "fig5_before_after")',
     '    ax.set_ylabel("Reward", fontsize=8)\n    ax.set_title("Before vs after training", fontsize=9, fontweight="bold", pad=8)\n    _save(fig, "fig5_before_after")'),
]

for old, new in replacements:
    if old in t:
        t = t.replace(old, new)
    else:
        # Show what didn't match
        print(f"WARNING: pattern not found:\n  {repr(old[:60])}...")

with open("nature_figures.py", "w", encoding="utf-8") as f:
    f.write(t)

# Compile check
try:
    compile(t, "nature_figures.py", "exec")
    print("nature_figures.py: OK")
except SyntaxError as e:
    print(f"Syntax error: {e}")

# Now fix param_figures.py
with open("param_figures.py", "r", encoding="utf-8") as f:
    t2 = f.read()

# Param figures need titles too
preplacements = [
    ('ax.set_ylabel("Last 50 mean reward",fontsize=8)\nax.set_ylim(0,480)',
     'ax.set_ylabel("Last 50 mean reward",fontsize=8)\nax.set_title("PPO learning rate effect",fontsize=9,fontweight="bold",pad=8)\nax.set_ylim(0,480)'),
    
    ('ax.set_ylabel("Last 100 mean reward",fontsize=8)\nax.set_ylim(0,550)',
     'ax.set_ylabel("Last 100 mean reward",fontsize=8)\nax.set_title("A2C n_steps effect",fontsize=9,fontweight="bold",pad=8)\nax.set_ylim(0,550)'),
    
    ('ax.set_ylabel("Mean reward",fontsize=8)\nax.set_ylim(0,550)',
     'ax.set_ylabel("Mean reward",fontsize=8)\nax.set_title("Default vs optimal A2C",fontsize=9,fontweight="bold",pad=8)\nax.set_ylim(0,550)'),
]

for old, new in preplacements:
    if old in t2:
        t2 = t2.replace(old, new)

with open("param_figures.py", "w", encoding="utf-8") as f:
    f.write(t2)

try:
    compile(t2, "param_figures.py", "exec")
    print("param_figures.py: OK")
except SyntaxError as e:
    print(f"Syntax error: {e}")

print("Done")
