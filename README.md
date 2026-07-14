# 基于强化学习的倒立摆控制（自定义任务环境）

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.1+-orange.svg)](https://stable-baselines3.readthedocs.io/)

## 项目简介

本项目是"AI赋能自动控制原理"课程的实验项目，使用强化学习算法（PPO和A2C）解决定制化CartPole（小车-倒立摆）系统的平衡控制任务。

**任务环境：** 基于OpenAI Gymnasium的CartPole-v1物理引擎，按任务描述定制：
- 终止角度：15°（而非默认12°）
- 失败惩罚：-10（而非默认0）
- 每步存活奖励：+1
- 最大步数：500步

**核心目标：** 通过强化学习让智能体自主学习控制策略，在定制环境中实现稳定倒立摆控制。

**主要成果：**
- ✅ PPO算法在200,000步内完美解决，20回合评估全部满分（500步），标准差为0
- ✅ 优化后的A2C（n_steps=64, lr=5e-4）评估均值493.4，达标率90%
- ✅ 与传统控制方法（LQR、规则控制）对比，RL方法展现出绝对优势

## 项目结构

```
task1_cartpole/
├── train_task.py                   # 训练/评估/可视化脚本
├── requirements.txt                # Python依赖包
├── README.md                       # 项目说明文档
│
├── src/                            # 源代码目录
│   ├── __init__.py                 # 包初始化
│   ├── task_env.py                 # 自定义CartPole环境（15°、-10惩罚）
│   └── baseline_controllers.py     # 传统控制基线（LQR+规则控制）
│
├── nature_figures.py              # Nature风格图表生成脚本
├── param_figures.py               # 参数验证图表生成脚本
│
├── models_task/                    # 训练好的模型（自动生成）
│   ├── ppo_cartpole_task.zip       # PPO模型（已收敛）
│   └── a2c_cartpole_task.zip       # A2C模型（接近收敛）
│
└── results_task/                   # 输出结果（自动生成）
    ├── fig1_ppo_training.png       # PPO训练曲线
    ├── fig2_a2c_training.png       # A2C训练曲线
    ├── fig3_method_comparison.png  # 控制方法对比柱状图
    ├── fig4_training_comparison.png# PPO vs A2C对比
    ├── fig5_before_after.png       # 训练前后效果对比
    ├── fig6_metrics_panel.png      # 多指标分析面板
    ├── fig_s1_ppo_lr.png           # PPO学习率对比
    ├── fig_s2_a2c_nsteps.png       # A2C n_steps对比
    ├── fig_s3_a2c_default_vs_optimal.png # A2C默认vs最优
    ├── *_train_log.json            # 训练日志
    └── baseline_comparison.json    # 传统控制基线数据
```

## 环境配置

### 系统要求

- **Python版本：** 3.8 或更高
- **操作系统：** Windows / macOS / Linux
- **GPU：** 可选（本项目默认使用CPU训练）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/zhaihuahua78/cartpole--.git
cd cartpole--

# 创建虚拟环境（推荐）
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 依赖包说明

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `stable-baselines3` | >= 2.1.0 | 强化学习算法实现（PPO、A2C） |
| `gymnasium` | >= 0.29.0 | CartPole物理引擎 |
| `matplotlib` | >= 3.7.0 | 数据可视化 |
| `numpy` | >= 1.24.0 | 数值计算 |

## 使用说明

### 快速开始

```bash
# 训练PPO（10万步）并评估
python train_task.py --train ppo

# 训练A2C（20万步）并评估
python train_task.py --train a2c

# 完整流程（训练PPO+A2C + 基线对比 + 图表）
python train_task.py --train both
```

### 关键参数验证

```bash
# 生成参数验证图表（PPO学习率对比 + A2C超参数扫描）
python param_figures.py
```

### 图表生成

```bash
# 生成所有Nature风格图表
python nature_figures.py
```

## 实验结果

### 性能对比（20回合评估）

| 方法 | 平均奖励 | 标准差 | 达标率(≥475) | 训练步数 |
|:----:|:--------:|:------:|:------------:|:--------:|
| 随机策略 | 16.1 | 6.8 | 0% | 0 |
| LQR控制器 | 44.8 | 7.4 | 0% | 0 |
| 规则控制器 | 215.8 | 29.1 | 0% | 0 |
| A2C（优化） | 493.4 | 20.8 | 90% | 200,000 |
| **PPO** | **500.0** | **0.0** | **100%** | **200,000** |

### 关键发现

- **PPO**：完美收敛，唯一兼具高能力（500分）和高稳定性（标准差0）的方法
- **A2C（优化后）**：经7种配置扫描，最优参数n_steps=64, lr=5e-4，评估均值493.4，达标率90%
- **规则控制器**：传统方法中最优（215.8分），无需训练
- **LQR控制器**：44.8分，线性化模型在±15°非线性范围内基本失效

## 算法说明

### PPO (Proximal Policy Optimization)

- 裁剪机制（clip_range=0.2）保证更新稳定性
- 关键参数：lr=3e-4, n_steps=2,048, γ=0.99
- 训练结果：280回合首次满分，后100回合均值500.0

### A2C (Advantage Actor-Critic)

- 同步演员-评论家架构，经7种配置扫描调优
- 最优参数：n_steps=64, lr=5e-4, ent_coef=0.02
- 训练结果：后100回合均值464.5，评估均值493.4

## 致谢

本项目基于以下开源库实现：
- [OpenAI Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
