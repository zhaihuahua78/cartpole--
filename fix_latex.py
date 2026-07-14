"""修复 LaTeX 中的角度符号"""
import os
os.chdir("D:/大二下/AI赋能自动控制原理/AI赋能自动控制原理/AI赋能自动控制原理/task1_cartpole")

with open("task1_report.tex", "r", encoding="utf-8") as f:
    t = f.read()

# 修复 \"circ -> $^\\circ$ 在数字后面时
# 文件中实际内容是 15^\circ（字母circ前是反斜杠）
t = t.replace("15^\\circ", "15$^\\circ$")

# 修复 \\pm 15^\\circ
t = t.replace("\\pm 15^\\circ", "$\\pm 15^\\circ$")

# 修复文件名描述中的
t = t.replace("（15^\\circ终止角、-10惩罚）", "（15$^\\circ$终止角、-10惩罚）")
t = t.replace("（15^\\circ终止角、-10失败惩罚）", "（15$^\\circ$终止角、-10失败惩罚）")
t = t.replace("（15^\\circ终止角", "（15$^\\circ$终止角")

with open("task1_report.tex", "w", encoding="utf-8") as f:
    f.write(t)

# 验证
count = t.count("^\\circ")
print(f"Remaining unescaped ^\\circ: {count}")
# 检查是否都已被 $ 包围
import re
problematic = re.findall(r'[^$]15\^\\\\circ[^$]', t)
print(f"Problematic (no dollar wrap): {len(problematic)}")
for p in problematic[:3]:
    print(f"  {repr(p)}")
print("Done")
