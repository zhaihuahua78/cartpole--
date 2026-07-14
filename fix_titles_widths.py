import os
os.chdir("D:/大二下/AI赋能自动控制原理/AI赋能自动控制原理/AI赋能自动控制原理/task1_cartpole")

# Fix param_figures.py titles
with open("param_figures.py", "r", encoding="utf-8") as f:
    t = f.read()

# Use a marker that doesn't have pipe character issues
t = t.replace("Figure S2 | A2C n_steps effect", "A2C n_steps effect")
t = t.replace("Figure S3 | Default vs optimal A2C", "Default vs optimal A2C")

with open("param_figures.py", "w", encoding="utf-8") as f:
    f.write(t)
print("param_figures.py fixed")

# Fix LaTeX widths
with open("task1_report.tex", "r", encoding="utf-8") as f:
    t2 = f.read()

t2 = t2.replace(
    r"\includegraphics[width=0.52\textwidth]{results_task/fig5_before_after.png}",
    r"\includegraphics[width=0.42\textwidth]{results_task/fig5_before_after.png}"
)
t2 = t2.replace(
    r"\includegraphics[width=0.62\textwidth]{results_task/fig3_method_comparison.png}",
    r"\includegraphics[width=0.50\textwidth]{results_task/fig3_method_comparison.png}"
)

with open("task1_report.tex", "w", encoding="utf-8") as f:
    f.write(t2)
print("LaTeX widths fixed")
