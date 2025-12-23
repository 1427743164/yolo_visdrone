import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil


def convert_visdrone_to_yolo(visdrone_root, save_root):
    """
    将VisDrone数据集转换为YOLOv11格式 [cite: 91]
    """
    # 类别映射：忽略0和11，保留常用的10类 [cite: 99]
    class_map = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
        6: 5, 7: 6, 8: 7, 9: 8, 10: 9
    }

    splits = {
        'VisDrone2019-DET-train': 'train',
        'VisDrone2019-DET-val': 'val',
        'VisDrone2019-DET-test-dev': 'test'
    }

    visdrone_root = Path(visdrone_root)
    save_root = Path(save_root)

    for split_dir, split_name in splits.items():
        src_img_dir = visdrone_root / split_dir / 'images'
        src_ann_dir = visdrone_root / split_dir / 'annotations'

        if not src_img_dir.exists():
            print(f"Directory not found: {src_img_dir}, skipping.")
            continue

        dst_img_dir = save_root / 'images' / split_name
        dst_lbl_dir = save_root / 'labels' / split_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {split_name} set...")
        img_files = list(src_img_dir.glob('*.jpg'))

        for img_path in tqdm(img_files):
            ann_path = src_ann_dir / (img_path.stem + '.txt')
            if not ann_path.exists(): continue

            try:
                with Image.open(img_path) as img:
                    w_img, h_img = img.size
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                continue

            yolo_labels = []
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8: continue

                    category = int(parts[5])
                    if category not in class_map: continue

                    bbox_left = int(parts[0])
                    bbox_top = int(parts[1])
                    bbox_w = int(parts[2])
                    bbox_h = int(parts[3])

                    # 归一化 xywh [cite: 139]
                    x_center = (bbox_left + bbox_w / 2) / w_img
                    y_center = (bbox_top + bbox_h / 2) / h_img
                    w_norm = bbox_w / w_img
                    h_norm = bbox_h / h_img

                    # 越界修正
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))

                    cid = class_map[category]
                    yolo_labels.append(f"{cid} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            if yolo_labels:
                with open(dst_lbl_dir / (img_path.stem + '.txt'), 'w') as f:
                    f.write('\n'.join(yolo_labels))
                shutil.copy(img_path, dst_img_dir / img_path.name)

# 使用示例
# convert_visdrone_to_yolo('D:/Datasets/VisDrone', 'D:/Projects/YOLOv11_VisDrone/datasets/VisDrone_YOLO')