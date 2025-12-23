import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from sklearn.metrics import confusion_matrix
import cv2


# ==========================================
# 1. 论文全局样式设置 (Times New Roman, 矢量图)
# ==========================================
def set_paper_style():
    # 尝试使用 seaborn-paper 风格
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('seaborn-paper')

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.unicode_minus': False,  # 解决负号显示问题
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.linewidth': 1.5,
        'pdf.fonttype': 42,  # 确保导出 PDF 字体可编辑
        'ps.fonttype': 42
    })


set_paper_style()


# ==========================================
# 2. 功能模块定义
# ==========================================

def plot_box_distribution(widths, heights, save_path='fig1_box_dist.pdf'):
    """
    绘制 BBox 宽高分布图 (散点 + 边缘直方图)
    调用时机：数据集准备阶段，分析 VisDrone 小目标分布。
    """
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4)

    # 主散点图
    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_scatter.scatter(widths, heights, alpha=0.3, s=15, c='#2b8cbe', edgecolors='none')
    ax_scatter.set_xlabel('Normalized Width')
    ax_scatter.set_ylabel('Normalized Height')
    ax_scatter.grid(True, linestyle='--', alpha=0.5)

    # 上方直方图
    ax_hist_x = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
    sns.histplot(x=widths, ax=ax_hist_x, color='#7bccc4', kde=True, element="step")
    ax_hist_x.axis('off')

    # 右侧直方图
    ax_hist_y = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)
    sns.histplot(y=heights, ax=ax_hist_y, color='#7bccc4', kde=True, element="step", orientation='horizontal')
    ax_hist_y.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[Success] Saved Box Distribution to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, save_path='fig2_confusion_matrix.pdf'):
    """
    绘制归一化混淆矩阵
    调用时机：验证/测试结束后 (val.py)，有了所有预测结果后。
    """
    cm = confusion_matrix(y_true, y_pred)
    # 归一化处理
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    plt.figure(figsize=(10, 8))
    # 使用 Blues 色系，annot=True 显示数值
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})

    plt.ylabel('Ground Truth', fontweight='bold')
    plt.xlabel('Prediction', fontweight='bold')
    plt.title('Confusion Matrix', fontweight='bold', pad=15)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[Success] Saved Confusion Matrix to {save_path}")
    plt.close()


def plot_zoom_in(img_array, boxes, zoom_region, save_path='fig3_zoom_in.pdf'):
    """
    绘制局部放大图
    调用时机：挑选 qualitative results (定性分析) 时，针对密集场景。
    zoom_region: [x, y, w, h] (原图坐标)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_array)

    # 绘制主图框 (绿色)
    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1.5, edgecolor='#00FF00', facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')

    # 创建放大子图 (zoom=3倍)
    # loc=2 (左上), bbox_to_anchor 控制子图具体位置
    axins = zoomed_inset_axes(ax, zoom=3, loc='upper right', bbox_to_anchor=(0.98, 0.98), bbox_transform=ax.transAxes)
    axins.imshow(img_array)

    # 绘制子图框
    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='#00FF00', facecolor='none')
        axins.add_patch(rect)

    # 设置子图视野
    x, y, w, h = zoom_region
    axins.set_xlim(x, x + w)
    axins.set_ylim(y + h, y)  # 图片坐标系Y轴向下
    axins.set_xticks([])
    axins.set_yticks([])

    # 连线
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[Success] Saved Zoom-in Visualization to {save_path}")
    plt.close()


def plot_heatmap_overlay(img_array, heatmap_data, save_path='fig4_heatmap.pdf'):
    """
    绘制 Attention/Grad-CAM 热力图叠加
    调用时机：模型可解释性分析。
    """
    plt.figure(figsize=(8, 6))

    # 1. 绘制底图
    plt.imshow(img_array)

    # 2. 处理热力图 (Resize 到图片大小 -> 归一化 -> 伪彩色)
    hm_resized = cv2.resize(heatmap_data, (img_array.shape[1], img_array.shape[0]))
    hm_norm = (hm_resized - hm_resized.min()) / (hm_resized.max() - hm_resized.min() + 1e-8)

    # 使用 jet 或 inferno 配色，alpha 控制透明度
    plt.imshow(hm_norm, cmap='jet', alpha=0.5)
    plt.axis('off')

    # 添加 colorbar
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label('Attention Weight')

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[Success] Saved Heatmap to {save_path}")
    plt.close()


# ==========================================
# 3. 主程序入口 (模拟数据运行)
# ==========================================
if __name__ == "__main__":
    print("Generating paper figures...")

    # --- 场景 1: 数据分布图 ---
    # 替换为: 读取你的 train.json 或 txt 标签文件
    dummy_w = np.random.beta(2, 10, 2000)  # 模拟小目标偏多的分布
    dummy_h = np.random.beta(2, 10, 2000)
    plot_box_distribution(dummy_w, dummy_h)

    # --- 场景 2: 混淆矩阵 ---
    # 替换为: val.py 跑出来的 pred_cls 和 target_cls 列表
    classes = ['Pedestrian', 'People', 'Bicycle', 'Car', 'Van', 'Truck', 'Tricycle', 'Awning-tricycle', 'Bus', 'Motor']
    y_true = np.random.randint(0, 10, 1000)
    y_pred = [y if np.random.rand() > 0.3 else np.random.randint(0, 10) for y in y_true]
    plot_confusion_matrix(y_true, y_pred, classes)

    # --- 场景 3: 局部放大图 ---
    # 替换为: 读取一张真实的测试图片 cv2.imread('test.jpg')
    # 并将 color 转换为 RGB: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sim_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)  # 模拟 1080p 图片
    # 替换为: 模型预测出的 boxes [x, y, w, h]
    sim_boxes = [[800, 500, 50, 60], [860, 520, 40, 50], [200, 200, 100, 100]]
    zoom_area = [750, 450, 200, 200]  # 指定要放大的密集区域
    plot_zoom_in(sim_img, sim_boxes, zoom_area)

    # --- 场景 4: 热力图 ---
    # 替换为: RT-DETR transformer 输出的 attention map (通常是 tensor)
    sim_heatmap = np.zeros((50, 50))
    # 模拟高亮中心
    cx, cy = 25, 25
    x = np.arange(0, 50)
    y = np.arange(0, 50)
    xx, yy = np.meshgrid(x, y)
    sim_heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 10 ** 2))

    plot_heatmap_overlay(sim_img, sim_heatmap)

    print("Done! Check the PDF files in your folder.")