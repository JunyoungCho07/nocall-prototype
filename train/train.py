"""YOLOv8 fine-tune on NoCall 5-class dataset.

Usage:
    uv run python train/train.py
"""
from pathlib import Path
from ultralytics import YOLO

YAML = Path(__file__).parent / "nocall.yaml"
EPOCHS = 80
IMGSZ = 640
BATCH = 8  # RTX 4050 6GB 기준 (640px yolov8s에서 16은 OOM 위험)


def main():
    model = YOLO("yolov8s.pt")  # small — n보다 정확, M5에서 충분
    results = model.train(
        data=str(YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project="runs/train",
        name="nocall",
        exist_ok=True,
        patience=20,        # 조기 종료
        augment=True,
        degrees=10,         # 실제 -45도 장착 후 각도 흔들림 보정
        scale=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
    )
    # best.pt → models/ 에 복사 (ultralytics 실제 저장 경로 사용)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    dest = Path(__file__).parent.parent / "models" / "nocall.pt"
    dest.parent.mkdir(exist_ok=True)
    import shutil
    shutil.copy(best, dest)
    print(f"모델 저장 완료: {dest}")


if __name__ == "__main__":
    main()
