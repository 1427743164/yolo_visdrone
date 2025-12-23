import matplotlib.pyplot as plt
import numpy as np

# 假设数据 (根据 PDF 中的预期结论 [cite: 335])
models = ['Baseline (YOLOv11)', '+ P2 Head', '+ P2 & SPD-Conv', '+ P2, SPD & NWD (Ours)']
ap_small = [18.5, 21.2, 22.8, 24.1] # AP_small 指标
ap_total = [28.0, 30.5, 31.8, 33.2] # mAP_50 指标

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, ap_small, width, label='AP_small (Tiny Objects)', color='#88c999')
rects2 = ax.bar(x + width/2, ap_total, width, label='mAP@0.5 (Overall)', color='#6f95ad')

ax.set_ylabel('Precision (%)', fontsize=12)
ax.set_title('Ablation Study on VisDrone Dataset', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend()
ax.set_ylim(0, 40)

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('ablation_study.png', dpi=300)
plt.show()