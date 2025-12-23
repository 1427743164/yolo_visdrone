from ultralytics import YOLO


def main():
    # 加载包含 SPD-Conv 和 P2 Head 的模型 [cite: 291]
    model = YOLO("yolov11-visdrone.yaml")

    # 加载预训练权重 (部分层会因架构改变而重新初始化) [cite: 295]
    try:
        model.load("yolo11n.pt")
    except KeyError:
        print("Pretrained weights loaded partially.")

    # 开始训练
    model.train(
        data="VisDrone.yaml",
        epochs=150,  # VisDrone 建议增加 epoch [cite: 301]
        imgsz=640,  # 显存允许建议设为 1280 [cite: 328]
        batch=4,  # P2 Head 显存占用大 [cite: 305]
        device=0,
        optimizer="AdamW",
        name="yolov11_visdrone_innovation",
        lr0=0.001,
        mosaic=1.0,
        copy_paste=0.3,  # 增加微小目标密度
        mixup=0.1  # 降低 mixup 避免特征模糊 [cite: 316]
    )


if __name__ == '__main__':
    main()