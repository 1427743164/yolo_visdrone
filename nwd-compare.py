import numpy as np
import matplotlib.pyplot as plt
import torch


# 引入之前定义的 wasserstein_loss 函数
def wasserstein_loss_sim(box1, box2, constant=12.5):
    # 简化版实现用于绘图
    b1_cx, b1_cy, b1_w, b1_h = box1
    b2_cx, b2_cy, b2_w, b2_h = box2
    w2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2 + ((b1_w - b2_w) / 2) ** 2 + ((b1_h - b2_h) / 2) ** 2
    return np.exp(-np.sqrt(w2) / constant)


def iou_loss_sim(box1, box2):
    # 简化版 IoU 计算 (假设 w, h 相同，仅移动中心)
    # box: cx, cy, w, h
    inter_w = max(0,
                  min(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2) - max(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2))
    inter_h = max(0,
                  min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2) - max(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2))
    inter_area = inter_w * inter_h
    union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return 1 - iou


# 模拟实验：一个 6x6 的微小目标，预测框从中心偏移 0 到 10 像素
pixels_shift = np.linspace(0, 10, 100)
nwd_vals = []
iou_vals = []

target_box = [50, 50, 6, 6]  # [cx, cy, w, h]

for shift in pixels_shift:
    pred_box = [50 + shift, 50 + shift, 6, 6]  # 向对角线偏移

    nwd_loss = 1 - wasserstein_loss_sim(target_box, pred_box)
    iou_loss = iou_loss_sim(target_box, pred_box)

    nwd_vals.append(nwd_loss)
    iou_vals.append(iou_loss)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(pixels_shift, iou_vals, label='IoU Loss (Standard)', linestyle='--', color='red', linewidth=2)
plt.plot(pixels_shift, nwd_vals, label='NWD Loss (Ours)', color='blue', linewidth=2)
plt.title('Loss Sensitivity to Pixel Shift for Tiny Objects (6x6 pixels)', fontsize=14)
plt.xlabel('Pixel Deviation (px)', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.axvline(x=6, color='gray', linestyle=':', label='Object Size Boundary')  # 目标尺寸边界
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('nwd_vs_iou_analysis.png', dpi=300)
plt.show()