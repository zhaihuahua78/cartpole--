import os
os.chdir("D:/大二下/AI赋能自动控制原理/AI赋能自动控制原理/AI赋能自动控制原理/task1_cartpole")

with open("nature_figures.py", "r", encoding="utf-8") as f:
    t = f.read()

# Remove any dangling continuation lines from removed set_title/suptitle
import re
# Remove lines that are just continuation arguments without a function call before them
t = re.sub(r'^\s+fontsize=9, fontweight="bold", pad=8\)\n', '', t, flags=re.MULTILINE)
t = re.sub(r'^\s+fontsize=9, fontweight="bold", y=1\.\d+\)\n', '', t, flags=re.MULTILINE)

with open("nature_figures.py", "w", encoding="utf-8") as f:
    f.write(t)

# Verify syntax
try:
    compile(t, "nature_figures.py", "exec")
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax error: {e}")
    # Show the problematic line
    lines = t.split("\n")
    print(f"Line {e.lineno}: {repr(lines[e.lineno-1])}")

print("Done")
