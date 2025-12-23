import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil


def convert_visdrone_to_yolo(visdrone_root, save_root):
    """
    将 VisDrone 数据集转换为 YOLOv11 格式
    主要修正：增加了 score 字段过滤，去除了官方标记为 'ignored' 的低质量框
    """
    # 类别映射：只保留 1-10 (YOLO index 0-9)
    # 0 (ignored regions) 和 11 (others) 通常不用于训练
    class_map = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
        6: 5, 7: 6, 8: 7, 9: 8, 10: 9
    }

    # 你的 VisDrone 文件夹名称可能不同，请确认
    splits = {
        'VisDrone2019-DET-train': 'train',
        'VisDrone2019-DET-val': 'val',
        'VisDrone2019-DET-test-dev': 'test'
    }

    visdrone_root = Path(visdrone_root)
    save_root = Path(save_root)

    # 确保输出目录存在
    if save_root.exists():
        print(f"Warning: 输出目录 {save_root} 已存在，转换将覆盖或合并文件。")

    for split_dir, split_name in splits.items():
        src_img_dir = visdrone_root / split_dir / 'images'
        src_ann_dir = visdrone_root / split_dir / 'annotations'

        if not src_img_dir.exists():
            print(f"Skipping {split_dir}: 目录不存在")
            continue

        # YOLO 标准目录结构: images/train, labels/train
        dst_img_dir = save_root / 'images' / split_name
        dst_lbl_dir = save_root / 'labels' / split_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔄 正在转换 {split_name} 集...")
        img_files = list(src_img_dir.glob('*.jpg'))

        for img_path in tqdm(img_files, desc=f"Converting {split_name}"):
            # 1. 复制图片 (为了保证数据独立性，建议复制；如果想省空间可以用软链接)
            dst_img_path = dst_img_dir / img_path.name
            if not dst_img_path.exists():
                shutil.copy(img_path, dst_img_path)

            # 2. 读取图片尺寸 (用于归一化)
            # 优化：try-except 捕获损坏图片
            try:
                with Image.open(img_path) as img:
                    w_img, h_img = img.size
            except Exception as e:
                print(f"❌ Error reading image {img_path}: {e}")
                continue

            # 3. 处理标签
            ann_path = src_ann_dir / (img_path.stem + '.txt')
            if not ann_path.exists():
                continue  # 没有标签的图片通常作为背景图（Negative Sample）保留，或者跳过

            yolo_labels = []
            with open(ann_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) < 8: continue

                    # VisDrone 格式: <x>,<y>,<w>,<h>,<score>,<category>,<truncation>,<occlusion>

                    # [关键修正] 检查 score (index 4)
                    # score 0: ignored, score 1: considered
                    score = int(parts[4])
                    if score == 0: continue  # 🚨 必须过滤掉 score=0 的框，否则会干扰模型！

                    category = int(parts[5])
                    if category not in class_map: continue  # 过滤掉 ignroed regions(0) 和 others(11)

                    bbox_left = int(parts[0])
                    bbox_top = int(parts[1])
                    bbox_w = int(parts[2])
                    bbox_h = int(parts[3])

                    # 归一化 xywh
                    x_center = (bbox_left + bbox_w / 2) / w_img
                    y_center = (bbox_top + bbox_h / 2) / h_img
                    w_norm = bbox_w / w_img
                    h_norm = bbox_h / h_img

                    # 越界修正 (Clamp 0-1)
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))

                    cid = class_map[category]
                    yolo_labels.append(f"{cid} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            # 写入 txt
            if yolo_labels:
                with open(dst_lbl_dir / (img_path.stem + '.txt'), 'w') as f:
                    f.write('\n'.join(yolo_labels))

    print(f"\n✅ 转换完成！数据保存在: {save_root}")
    print("请确保你的 .yaml 文件指向该目录。")


# --- 请修改下面的路径为你自己的路径 ---
if __name__ == "__main__":
    # 原始 VisDrone 路径 (包含 VisDrone2019-DET-train 等文件夹)
    ORIGIN_DIR = r"datasets"

    # 转换后的保存路径 (YOLO 格式)
    OUTPUT_DIR = r"datasets/VisDrone_YOLO"

    convert_visdrone_to_yolo(ORIGIN_DIR, OUTPUT_DIR)