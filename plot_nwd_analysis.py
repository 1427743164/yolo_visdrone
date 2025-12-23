import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置论文格式
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


def calculate_iou(box1, box2):
    """计算标准 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)


def calculate_nwd(box1, box2, constant=12.8):
    """
    计算 NWD (Normalized Wasserstein Distance)
    box: [x1, y1, x2, y2]
    """
    # 将 xyxy 转换为 center_x, center_y, w, h
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    cx1, cy1 = box1[0] + w1 / 2, box1[1] + h1 / 2
    cx2, cy2 = box2[0] + w2 / 2, box2[1] + h2 / 2

    # Wasserstein Distance calculation (简化版, 假设它是 2D Gaussian)
    # W2^2 = ||m1-m2||^2 + Tr(...)
    # 对于轴对齐框:
    # W2^2 = (cx1-cx2)^2 + (cy1-cy2)^2 + ((w1-w2)/2)^2 + ((h1-h2)/2)^2
    w2_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) / 2) ** 2 + ((h1 - h2) / 2) ** 2

    # NWD = exp( - sqrt(W2^2) / C )
    nwd = np.exp(-np.sqrt(w2_dist_sq) / constant)
    return nwd


def plot_sensitivity_curve():
    # 模拟一个小目标: 16x16 像素
    base_box = [100, 100, 116, 116]

    shifts = np.arange(0, 20, 0.5)  # 偏移量从 0 到 20 像素
    ious = []
    nwds = []

    for s in shifts:
        # 向右平移 s 个像素
        shifted_box = [100 + s, 100, 116 + s, 116]

        ious.append(calculate_iou(base_box, shifted_box))
        nwds.append(calculate_nwd(base_box, shifted_box, constant=12.0))

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(7, 5))

    # 绘制 IoU 曲线 (红色, 虚线)
    ax.plot(shifts, ious, label='IoU Metric', color='#d62728', linestyle='--', linewidth=2.5)

    # 绘制 NWD 曲线 (蓝色, 实线)
    ax.plot(shifts, nwds, label='NWD Metric (Ours)', color='#1f77b4', linestyle='-', linewidth=2.5)

    # 标注关键区域
    plt.axvline(x=16, color='gray', linestyle=':', alpha=0.5)
    plt.text(16.5, 0.6, 'No Overlap\n(IoU=0)', fontsize=10, color='gray')

    ax.set_xlabel('Pixel Deviation (Shift)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Similarity Score', fontsize=12, fontweight='bold')
    ax.set_title('Sensitivity Analysis: NWD vs. IoU on Tiny Objects', fontsize=14, pad=15)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_nwd_sensitivity.pdf')
    print("Saved NWD analysis chart.")
    plt.show()


if __name__ == '__main__':
    plot_sensitivity_curve()