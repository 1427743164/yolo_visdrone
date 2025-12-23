import matplotlib.pyplot as plt
import numpy as np

# 风格设置
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


def plot_ablation_study():
    # --- 数据准备 (替换为你论文的真实数据) ---
    methods = ['Baseline\n(RT-DETR)', '+ Wavelet', '+ NWD Loss', '+ Both\n(Ours)']
    map_scores = [35.2, 36.8, 37.5, 38.9]  # 对应的 mAP 50

    # 配色：前几个用灰色，最后一个(你的方法)用高亮色
    colors = ['#d9d9d9', '#bdbdbd', '#969696', '#2b8cbe']

    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制柱状图
    bars = ax.bar(methods, map_scores, color=colors, width=0.6, edgecolor='black', linewidth=1)

    # 设置 Y 轴范围 (为了让差异看起来明显，不要从 0 开始，从 Baseline 附近开始)
    min_score = min(map_scores) - 2
    max_score = max(map_scores) + 2
    ax.set_ylim(min_score, max_score)

    # 添加数值标签和增长箭头
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # 在柱子上方写 mAP 数值
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.2,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 绘制增长箭头 (除了第一个)
        if i > 0:
            prev_height = map_scores[i - 1]
            gain = height - prev_height
            # 只有当有提升时才画
            ax.annotate(f'+{gain:.1f}',
                        xy=(bar.get_x(), (height + prev_height) / 2),
                        xytext=(-15, 0), textcoords='offset points',
                        color='#e6550d', fontweight='bold', fontsize=10,
                        arrowprops=dict(arrowstyle='->', color='#e6550d'))

    ax.set_ylabel('mAP@0.5 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Contribution of Each Module', fontsize=14, pad=20)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('fig_ablation_bar.pdf')
    print("Saved Ablation chart.")
    plt.show()


if __name__ == '__main__':
    plot_ablation_study()