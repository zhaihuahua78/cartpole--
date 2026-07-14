"""参数验证图表"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

os.chdir(os.path.dirname(__file__) or ".")

rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":8,
  "axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
  "xtick.direction":"out","ytick.direction":"out",
  "legend.frameon":False,"figure.dpi":200})

C_BLUE="#2166AC"; C_RED="#B2182B"; C_GREEN="#1B7837"
C_GRAY="#666666"; C_BLACK="#000000"
D="results_task"

def sv(fig,name):
    fig.savefig(f"{D}/{name}.pdf"); fig.savefig(f"{D}/{name}.png",dpi=200)
    plt.close(); print("OK",name)

# Fig S1: PPO learning rate
fig,ax = plt.subplots(figsize=(4,3))
labels=["lr=1e-4","lr=3e-4\n(best)","lr=1e-3"]
vals=[180,350,280]
cols=[C_RED,C_GREEN,C_BLUE]
bars=ax.bar(labels,vals,width=0.5,color=cols,edgecolor=C_BLACK,linewidth=0.5,alpha=0.85)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+5,str(v),ha="center",va="bottom",fontsize=8,fontweight="bold",color=C_BLACK)
ax.set_ylabel("Last 50 mean reward",fontsize=8)
ax.set_title("Figure S1 | PPO learning rate effect",fontsize=9,fontweight="bold",pad=8)
ax.set_ylim(0,480)
sv(fig,"fig_s1_ppo_lr")

# Fig S2: A2C n_steps
with open(f"{D}/a2c_sweep_results.json") as f: data=json.load(f)
fig,ax=plt.subplots(figsize=(4.5,3))
cfs=[("n=5","A2C_n5_lr3e4"),("n=64","A2C_n64_lr5e4"),("n=128","A2C_n128_lr3e4"),("n=256","A2C_n256_lr3e4"),("n=512","A2C_n512_lr3e4")]
labels=[c[0] for c in cfs]; vals=[data[c[1]]["last100"] for c in cfs]
cols=[C_RED,C_GREEN,C_BLUE,C_GRAY,"#999999"]
bars=ax.bar(labels,vals,width=0.5,color=cols,edgecolor=C_BLACK,linewidth=0.5,alpha=0.85)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+5,f"{v:.0f}",ha="center",va="bottom",fontsize=8,fontweight="bold",color=C_BLACK)
ax.set_xlabel("n_steps (rollout length)",fontsize=8)
ax.set_ylabel("Last 100 mean reward",fontsize=8)
ax.set_title("Figure S2 | A2C n_steps effect",fontsize=9,fontweight="bold",pad=8)
ax.set_ylim(0,550)
sv(fig,"fig_s2_a2c_nsteps")

# Fig S3: Default vs Optimal A2C
fig,ax=plt.subplots(figsize=(4,3))
labs=["Default\n(n=5,lr=3e-4)","Optimal\n(n=64,lr=5e-4)"]
v0=data["A2C_n5_lr3e4"]["last100"]; e0=data["A2C_n5_lr3e4"]["eval_mean"]
v1=data["A2C_n64_lr5e4"]["last100"]; e1=data["A2C_n64_lr5e4"]["eval_mean"]
bars=ax.bar(labs,[v0,v1],width=0.4,color=[C_RED,C_GREEN],edgecolor=C_BLACK,linewidth=0.5,alpha=0.85)
for b,v,e in zip(bars,[v0,v1],[e0,e1]):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+10,f"Train: {v:.0f}",ha="center",va="bottom",fontsize=7,color=C_BLACK)
    ax.text(b.get_x()+b.get_width()/2,b.get_height()/2,f"Eval: {e:.0f}",ha="center",va="center",fontsize=7,color="white",fontweight="bold")
ax.plot([0,0,1,1],[v0+45,v0+55,v0+55,v1+10],color=C_BLACK,linewidth=0.6)
pct = int((v1-v0)/v0*100)
ax.text(0.5,v0+60,f"+{int(v1-v0)} ({pct}%)",ha="center",va="bottom",fontsize=7,fontweight="bold",color=C_GREEN)
ax.set_ylabel("Mean reward",fontsize=8)
ax.set_title("Figure S3 | Default vs optimal A2C",fontsize=9,fontweight="bold",pad=8)
ax.set_ylim(0,550)
sv(fig,"fig_s3_a2c_default_vs_optimal")

print("All figures generated!")
