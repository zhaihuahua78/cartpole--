import os, re
os.chdir("D:/大二下/AI赋能自动控制原理/AI赋能自动控制原理/AI赋能自动控制原理/task1_cartpole")

# Fix nature_figures.py
with open("nature_figures.py", "r", encoding="utf-8") as f:
    t = f.read()

remove = [
    'ax.set_title("PPO training curve"',
    'ax.set_title("A2C training curve"',
    'ax.set_title("Control method comparison"',
    'ax1.set_title("PPO"',
    'ax2.set_title("A2C"',
    'ax.set_title("Before vs after training"',
    'fig.suptitle("Training process comparison"',
    'fig.suptitle("Multi-metric training analysis"',
]

for pattern in remove:
    # Find the full line containing this pattern and remove it
    for line in t.split("\n"):
        if pattern in line:
            t = t.replace(line + "\n", "")
            break

with open("nature_figures.py", "w", encoding="utf-8") as f:
    f.write(t)

# Count remaining
remaining = [l for l in t.split("\n") if "set_title" in l or "suptitle" in l]
print(f"nature_figures.py: {len(remaining)} remaining title calls")

# Fix param_figures.py
with open("param_figures.py", "r", encoding="utf-8") as f:
    t2 = f.read()

remove2 = [
    'ax.set_title("PPO learning rate effect"',
    'ax.set_title("A2C n_steps effect"',
    'ax.set_title("Default vs optimal A2C"',
]

for pattern in remove2:
    for line in t2.split("\n"):
        if pattern in line:
            t2 = t2.replace(line + "\n", "")
            break

with open("param_figures.py", "w", encoding="utf-8") as f:
    f.write(t2)

remaining2 = [l for l in t2.split("\n") if "set_title" in l]
print(f"param_figures.py: {len(remaining2)} remaining title calls")
print("Done")
