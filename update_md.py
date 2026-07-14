"""更新 Markdown 报告：添加参数验证图 + 重写附录"""
import os
os.chdir("D:/大二下/AI赋能自动控制原理/AI赋能自动控制原理/AI赋能自动控制原理/task1_cartpole")

with open("task1_report.md", "r", encoding="utf-8") as f:
    txt = f.read()

# Add parameter figures after 2.3.2
old = "**关键发现：** n_steps=64 最优"
new = old + '\n\n![PPO学习率对比](results_task/fig_s1_ppo_lr.png)\n*Figure S1：PPO学习率对比。lr=3e-4在收敛速度和稳定性之间取得最佳平衡。*\n\n![A2C n_steps对比](results_task/fig_s2_a2c_nsteps.png)\n*Figure S2：A2C不同n_steps配置对比。n_steps=64最优。*\n\n![默认vs最优A2C](results_task/fig_s3_a2c_default_vs_optimal.png)\n*Figure S3：A2C默认配置与最优配置对比。优化后后100均值从41.2提升至433.5（+952%）。*'
txt = txt.replace(old, new)

# Rewrite appendix
old_app = "**代码仓库：** https://github.com/zhaihuahua78/cartpole--\n\n---\n\n*报告完成日期：2026年7月*"
new_app = "## 5 附录\n\n本项目代码已开源至GitHub，可从以下链接获取完整代码：\n\nhttps://github.com/zhaihuahua78/cartpole--\n\n代码仓库包含：\n- `src/task_env.py`：符合任务描述的CartPole自定义环境\n- `train_task.py`：完整训练、评估和可视化脚本\n- `nature_figures.py`：Nature期刊风格图表生成脚本\n- `param_figures.py`：参数验证图表生成脚本\n- `models_task/`：训练好的PPO和A2C模型文件\n- `results_task/`：所有实验数据、训练日志和图表\n\n使用前请安装依赖：`pip install -r requirements.txt`\n\n---\n\n*报告完成日期：2026年7月*"
txt = txt.replace(old_app, new_app)

with open("task1_report.md", "w", encoding="utf-8") as f:
    f.write(txt)
print("OK: Markdown report updated")
