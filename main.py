#!/usr/bin/env python
"""
任务一：基于强化学习的倒立摆控制 — 统一入口
用法:
    python main.py train --algo dqn          # 训练DQN
    python main.py train --algo ppo          # 训练PPO
    python main.py train --algo both         # 训练DQN和PPO
    python main.py eval --algo dqn           # 评估DQN模型
    python main.py eval --algo both          # 评估两个模型
    python main.py sweep --algo dqn          # DQN参数搜索
    python main.py sweep --algo both         # 两个算法参数搜索
    python main.py baseline                  # 传统控制基线
    python main.py plot                      # 生成可视化图表
    python main.py video --algo dqn          # 录制视频
    python main.py all                       # 执行全部流程
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    parser = argparse.ArgumentParser(description="任务一：基于强化学习的倒立摆控制")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # train
    train_parser = subparsers.add_parser("train", help="训练智能体")
    train_parser.add_argument("--algo", choices=["dqn", "ppo", "both"], default="both")
    train_parser.add_argument("--timesteps", type=int, default=50000)

    # eval
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument("--algo", choices=["dqn", "ppo", "both"], default="both")
    eval_parser.add_argument("--episodes", type=int, default=10)

    # sweep
    sweep_parser = subparsers.add_parser("sweep", help="关键参数搜索")
    sweep_parser.add_argument("--algo", choices=["dqn", "ppo", "both"], default="both")

    # baseline
    subparsers.add_parser("baseline", help="传统控制基线（LQR）")

    # plot
    subparsers.add_parser("plot", help="生成可视化图表")

    # video
    video_parser = subparsers.add_parser("video", help="录制演示视频")
    video_parser.add_argument("--algo", choices=["dqn", "ppo", "both"], default="both")

    # all
    subparsers.add_parser("all", help="执行全部流程")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    from src.env_analysis import print_env_info
    from src.train_dqn import train_dqn
    from src.train_ppo import train_ppo
    from src.eval_agent import evaluate
    from src.param_sweep_dqn import sweep_dqn
    from src.param_sweep_ppo import sweep_ppo
    from src.baseline_controllers import run_baseline
    from src.plot_results import plot_all
    from src.record_video import record

    if args.command == "all" or args.command == "train":
        print_env_info()
        if args.command == "all" or args.algo in ("dqn", "both"):
            train_dqn(timesteps=args.timesteps if hasattr(args, "timesteps") else 50000)
        if args.command == "all" or args.algo in ("ppo", "both"):
            train_ppo(timesteps=args.timesteps if hasattr(args, "timesteps") else 50000)

    if args.command == "eval" or args.command == "all":
        n = args.episodes if hasattr(args, "episodes") else 10
        if args.command == "all" or args.algo in ("dqn", "both"):
            evaluate("dqn", n_episodes=n)
        if args.command == "all" or args.algo in ("ppo", "both"):
            evaluate("ppo", n_episodes=n)

    if args.command == "sweep" or args.command == "all":
        if args.command == "all" or args.algo in ("dqn", "both"):
            sweep_dqn()
        if args.command == "all" or args.algo in ("ppo", "both"):
            sweep_ppo()

    if args.command == "baseline" or args.command == "all":
        run_baseline()

    if args.command == "video" or args.command == "all":
        if args.command == "all" or args.algo in ("dqn", "both"):
            record("dqn")
        if args.command == "all" or args.algo in ("ppo", "both"):
            record("ppo")

    if args.command == "plot" or args.command == "all":
        plot_all()

    print("\n✅ 全部流程完成！")


if __name__ == "__main__":
    main()
