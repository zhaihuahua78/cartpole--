# 基于强化学习的倒立摆控制

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.1+-orange.svg)](https://stable-baselines3.readthedocs.io/)

## 项目简介

本项目是"AI赋能自动控制原理"课程的实验项目，使用强化学习算法（DQN和PPO）解决经典控制问题——CartPole（小车-倒立摆）系统的平衡控制。

**核心目标：** 通过强化学习让智能体自主学会控制一个小车-倒立摆系统，使摆杆始终保持直立状态。

**主要成果：**
- ✅ PPO算法在50,000步内成功解决CartPole-v1，评估10回合全部满分（500步）
- ✅ DQN算法在150,000步内训练完成，最高可达满分
- ✅ 与传统控制方法（LQR、规则控制）对比，RL方法展现出明显优势

## 项目结构

```
task1_cartpole/
├── main.py                          # 统一入口脚本（支持所有功能）
├── requirements.txt                 # Python依赖包
├── README.md                        # 项目说明文档
├── task1_report.md                  # 实验报告（完整文档）
│
├── src/                             # 源代码目录
│   ├── __init__.py                  # 包初始化
│   ├── env_analysis.py              # 环境空间分析与可视化
│   ├── env_demo.py                  # 环境演示（随机策略）
│   ├── train_dqn.py                 # DQN算法训练
│   ├── train_ppo.py                 # PPO算法训练
│   ├── eval_agent.py                # 模型评估
│   ├── param_sweep_dqn.py           # DQN参数搜索实验
│   ├── param_sweep_ppo.py           # PPO参数搜索实验
│   ├── baseline_controllers.py      # 传统控制基线（LQR+规则控制）
│   ├── custom_reward_env.py         # 自定义奖励函数实验
│   ├── plot_results.py              # 可视化绑图
│   └── record_video.py              # 录制演示视频
│
├── models/                          # 训练好的模型（自动生成）
│   ├── dqn_cartpole.zip             # DQN模型文件
│   └── ppo_cartpole.zip             # PPO模型文件
│
└── results/                         # 输出结果（自动生成）
    ├── training_curves.png          # 训练奖励曲线
    ├── baseline_comparison.png      # 控制方法对比图
    ├── before_after_comparison.png  # 训练前后对比图
    ├── ppo_demo.gif                 # PPO演示视频
    ├── env_demo_random.gif          # 随机策略演示
    ├── dqn_train_log.json           # DQN训练日志
    ├── ppo_train_log.json           # PPO训练日志
    └── reward_comparison.json       # 奖励函数对比结果
```

## 环境配置

### 系统要求

- **Python版本：** 3.8 或更高
- **操作系统：** Windows / macOS / Linux
- **GPU：** 可选（本项目默认使用CPU训练，如需GPU加速需安装CUDA）

### 安装步骤

#### 1. 克隆或下载项目

```bash
# 方式1：克隆仓库（替换为实际仓库地址）
git clone https://github.com/your-username/cartpole-rl-control.git
cd cartpole-rl-control/task1_cartpole

# 方式2：直接下载ZIP文件并解压
```

#### 2. 创建虚拟环境（推荐）

```bash
# 使用conda创建虚拟环境
conda create -n cartpole python=3.10 -y
conda activate cartpole

# 或者使用venv
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖包说明

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `stable-baselines3` | >= 2.1.0 | 强化学习算法实现（DQN、PPO） |
| `gymnasium` | >= 0.29.0 | 强化学习环境（CartPole-v1） |
| `matplotlib` | >= 3.7.0 | 可视化绑图 |
| `numpy` | >= 1.24.0 | 数值计算 |
| `imageio` | >= 2.31.0 | GIF视频生成 |

## 使用说明

### 快速开始

```bash
# 执行全部流程（训练+评估+可视化）
python main.py all
```

### 详细功能

#### 1. 训练模型

```bash
# 训练DQN和PPO（默认50000步）
python main.py train --algo both

# 仅训练DQN，指定训练步数
python main.py train --algo dqn --timesteps 100000

# 仅训练PPO
python main.py train --algo ppo
```

#### 2. 评估模型

```bash
# 评估两个模型（默认10回合）
python main.py eval --algo both

# 评估PPO，指定回合数
python main.py eval --algo ppo --episodes 20
```

#### 3. 参数搜索

```bash
# 对DQN进行参数搜索
python main.py sweep --algo dqn

# 对PPO进行参数搜索
python main.py sweep --algo ppo
```

#### 4. 传统控制对比

```bash
# 运行传统控制基线（随机策略、规则控制、LQR）
python main.py baseline
```

#### 5. 生成可视化图表

```bash
# 生成所有图表
python main.py plot
```

#### 6. 录制演示视频

```bash
# 录制PPO演示视频
python main.py video --algo ppo

# 录制DQN演示视频
python main.py video --algo dqn
```

## 数据说明

### 输入数据

本项目使用OpenAI Gymnasium提供的CartPole-v1环境，无需额外输入数据。

**CartPole-v1环境参数：**
- 状态空间：4维连续向量 [小车位置, 小车速度, 摆杆角度, 摆杆角速度]
- 动作空间：离散2个动作 [0:左移, 1:右移]
- 最大回合步数：500步
- 解决条件：连续100回合平均奖励 ≥ 475

### 输出数据

训练过程中自动生成以下数据到 `results/` 目录：

| 文件 | 格式 | 说明 |
|------|------|------|
| `*_train_log.json` | JSON | 训练日志（每回合奖励和步数） |
| `*_eval_result.json` | JSON | 评估结果（20回合详细数据） |
| `training_curves.png` | PNG | DQN与PPO训练曲线对比 |
| `baseline_comparison.png` | PNG | 5种控制方法对比柱状图 |
| `before_after_comparison.png` | PNG | 训练前后控制效果对比 |
| `*.gif` | GIF | 智能体运行演示视频 |

## 算法说明

### DQN (Deep Q-Network)

基于值函数的深度强化学习算法，通过神经网络近似Q值函数。

**核心特性：**
- 经验回放缓冲区：50,000
- 目标网络更新间隔：500步
- 探索策略：ε-greedy（从1.0衰减到0.02）

**训练结果：**
- 训练步数：150,000
- 最后100回合均值：129.7
- 评估均值（20回合）：152.1 ± 155.2
- 最高奖励：500

### PPO (Proximal Policy Optimization)

基于策略梯度的深度强化学习算法，通过裁剪机制保证训练稳定性。

**核心特性：**
- 每次更新步数：2,048
- 训练轮数：10 epochs
- 裁剪范围：0.2

**训练结果：**
- 训练步数：50,000
- 最后100回合均值：497.4
- 评估均值（20回合）：500.0 ± 0.0
- 解决回合：345回合起持续满分

### 传统控制方法

**LQR控制器：**
- 基于线性化模型的最优控制
- 求解代数Riccati方程获得反馈增益
- 平均奖励：41步

**规则控制器：**
- 基于角度阈值的简单反馈控制
- 根据摆杆倾斜方向施加反向力
- 平均奖励：226步

## 实验结果

### 性能对比

| 方法 | 平均奖励 | 是否解决 |
|------|---------|----------|
| 随机策略 | 21 | ❌ |
| LQR控制器 | 41 | ❌ |
| 规则控制器 | 226 | ❌ |
| DQN | 152.1 ± 155.2 | ❌ |
| **PPO** | **500** | ✅ |

### 训练曲线

![训练曲线](results/training_curves.png)
*DQN与PPO训练奖励曲线对比*

### 控制方法对比

![基线对比](results/baseline_comparison.png)
*5种控制方法综合对比*

## 常见问题

### Q1: 训练速度很慢怎么办？

A1: 可以尝试以下方法：
- 减少训练步数：`--timesteps 30000`
- 使用GPU加速（需要安装CUDA和PyTorch GPU版本）
- 调整超参数（如增大`learning_rate`）

### Q2: DQN训练效果不好怎么办？

A2: DQN在CartPole上收敛较慢是正常的，建议：
- 增加训练步数到200,000+
- 调整`exploration_fraction`参数
- 直接使用PPO算法（效果更好）

### Q3: 如何在自己的环境中使用？

A3: 可以参考 `custom_reward_env.py` 中的实现，通过继承 `gym.Wrapper` 类来包装环境并修改奖励函数。

### Q4: 模型文件在哪里？

A4: 训练好的模型保存在 `models/` 目录下，格式为 `.zip`，可以通过 `stable-baselines3` 直接加载。

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 统一入口脚本，支持所有功能的命令行调用 |
| `src/env_analysis.py` | 打印CartPole环境的状态空间和动作空间信息 |
| `src/env_demo.py` | 使用随机策略演示环境，生成初始状态截图和GIF |
| `src/train_dqn.py` | DQN算法训练实现，包含奖励追踪回调 |
| `src/train_ppo.py` | PPO算法训练实现，包含奖励追踪回调 |
| `src/eval_agent.py` | 加载训练好的模型进行评估，输出性能指标 |
| `src/param_sweep_dqn.py` | DQN关键参数搜索（学习率、探索比例） |
| `src/param_sweep_ppo.py` | PPO关键参数搜索（学习率、裁剪范围） |
| `src/baseline_controllers.py` | 传统控制方法实现（LQR、规则控制、随机策略） |
| `src/custom_reward_env.py` | 自定义密集奖励函数设计与对比实验 |
| `src/plot_results.py` | 生成所有可视化图表（训练曲线、对比图等） |
| `src/record_video.py` | 录制智能体运行的GIF视频和关键帧截图 |

## 扩展阅读

- [Gymnasium文档](https://gymnasium.farama.org/) - 强化学习环境库
- [Stable-Baselines3文档](https://stable-baselines3.readthedocs.io/) - 强化学习算法库
- [CartPole环境说明](https://gymnasium.farama.org/environments/classic_control/cart_pole/) - CartPole-v1详细文档

## 致谢

本项目基于以下开源库实现：
- [OpenAI Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)

## 许可证

本项目仅供学习和研究使用。如需商业使用，请咨询相关授权。

---

**报告完成日期：** 2025年7月
