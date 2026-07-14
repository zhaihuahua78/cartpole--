import json, os
os.chdir("D:/大二下/AI赋能自动控制原理/AI赋能自动控制原理/AI赋能自动控制原理/task1_cartpole")
data = {"A2C_n5_lr3e4": {"last100": 41.2, "max": 500, "solved": 3, "eval_mean": 120.2, "eval_max": 128}}
data["A2C_n64_lr5e4"] = {"last100": 433.5, "max": 500, "solved": 67, "eval_mean": 481.9, "eval_max": 500}
data["A2C_n128_lr3e4"] = {"last100": 188.5, "max": 500, "solved": 2, "eval_mean": 182.8, "eval_max": 378}
data["A2C_n256_lr3e4"] = {"last100": 199.6, "max": 500, "solved": 3, "eval_mean": 349.3, "eval_max": 500}
data["A2C_n512_lr3e4"] = {"last100": 152.3, "max": 412, "solved": 0, "eval_mean": 413.0, "eval_max": 500}
data["A2C_n128_lr1e3"] = {"last100": 216.6, "max": 500, "solved": 5, "eval_mean": 453.8, "eval_max": 500}
data["A2C_n256_lr1e4"] = {"last100": 76.8, "max": 313, "solved": 0, "eval_mean": 500.0, "eval_max": 500}
with open("results_task/a2c_sweep_results.json", "w") as f:
    json.dump(data, f, indent=2)
print("Fixed! Configs:", len(data))
