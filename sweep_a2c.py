"""A2C 超参数扫描脚本"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8")

from src.task_env import make_task_env
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

RESULTS_DIR = "results_task"

class Tracker(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self._r = 0; self._l = 0
    def _on_step(self):
        self._r += self.locals["rewards"][0]
        self._l += 1
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._r)
            self.episode_lengths.append(self._l)
            self._r = 0; self._l = 0
        return True

def make_env():
    return make_task_env()

configs = [
    {"n_steps": 256, "lr": 3e-4, "ent_coef": 0.01, "label": "A2C_n256_lr3e4"},
    {"n_steps": 128, "lr": 3e-4, "ent_coef": 0.01, "label": "A2C_n128_lr3e4"},
    {"n_steps": 64,  "lr": 5e-4, "ent_coef": 0.02, "label": "A2C_n64_lr5e4"},
    {"n_steps": 128, "lr": 1e-3, "ent_coef": 0.005, "label": "A2C_n128_lr1e3"},
    {"n_steps": 5,   "lr": 3e-4, "ent_coef": 0.01, "label": "A2C_n5_lr3e4"},
    {"n_steps": 256, "lr": 1e-4, "ent_coef": 0.02, "label": "A2C_n256_lr1e4"},
    {"n_steps": 512, "lr": 3e-4, "ent_coef": 0.01, "label": "A2C_n512_lr3e4"},
]

results = {}
for cfg in configs:
    print(f"\n--- {cfg['label']} ---")
    env = DummyVecEnv([make_env])
    tracker = Tracker()

    model = A2C("MlpPolicy", env,
                learning_rate=cfg["lr"],
                n_steps=cfg["n_steps"],
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=cfg["ent_coef"],
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=0, seed=42, device="cpu")

    model.learn(total_timesteps=100000, callback=tracker)

    r = tracker.episode_rewards
    last100 = np.mean(r[-100:]) if len(r) >= 100 else np.mean(r)
    max_r = max(r)
    solved = sum(1 for v in r if v >= 475)
    print(f"  Episodes: {len(r)}, Last100: {last100:.1f}, Max: {max_r:.0f}, Solved: {solved}")

    # Quick eval
    eval_r = []
    for ep in range(10):
        env_e = make_task_env()
        obs, _ = env_e.reset(seed=ep)
        done = False
        total = 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, _ = env_e.step(int(a))
            done = term or trunc
            total += rew
        eval_r.append(total)
        env_e.close()
    print(f"  Eval(10): mean={np.mean(eval_r):.1f}, max={max(eval_r):.0f}")

    results[cfg["label"]] = {
        "last100": last100, "max": max_r, "solved": solved,
        "eval_mean": np.mean(eval_r), "eval_max": max(eval_r)
    }
    env.close()

# Print summary
print("\n" + "=" * 70)
print(f"{'Config':<20} {'Last100':>8} {'Max':>6} {'Solved':>7} {'EvalMean':>9} {'EvalMax':>8}")
print("-" * 70)
for label, res in sorted(results.items(), key=lambda x: -x[1]["last100"]):
    print(f"{label:<20} {res['last100']:>8.1f} {res['max']:>6.0f} {res['solved']:>7} {res['eval_mean']:>9.1f} {res['eval_max']:>8.0f}")

# Save results
with open(f"{RESULTS_DIR}/a2c_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to", f"{RESULTS_DIR}/a2c_sweep_results.json")
