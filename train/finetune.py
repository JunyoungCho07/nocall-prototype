"""기존 모델(models/nocall.pt)에서 이어서 fine-tune.

목적:
    웹캠 데이터가 추가된 학습 세트로 기존 가중치를 미세조정한다.
    처음부터 학습(train.py)하지 않고 이어 학습하므로 빠르고, 실제 환경 성능이 오른다.

사용법:
    1. capture.py → label_tool.py → add_webcam.py 를 먼저 끝낼 것
    2. uv run python train/finetune.py

GPU 자동 사용(RTX 4050). 완료 시 models/nocall.pt 갱신.
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

YAML = Path(__file__).parent / "nocall.yaml"
BASE = Path(__file__).parent.parent / "models" / "nocall.pt"  # 이어 학습할 기존 가중치
EPOCHS = 40
IMGSZ = 640
BATCH = 8  # RTX 4050 6GB 기준


def main():
    start = str(BASE) if BASE.exists() else "yolov8s.pt"
    print(f"이어 학습 시작점: {start}")
    model = YOLO(start)
    model.train(
        data=str(YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project="runs/train",
        name="nocall_ft",
        exist_ok=True,
        patience=15,
        lr0=0.005,        # fine-tune은 낮은 학습률로 기존 지식 유지
        augment=True,
        degrees=10,
        scale=0.4,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    dest = Path(__file__).parent.parent / "models" / "nocall.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best, dest)
    print(f"모델 갱신 완료: {dest}")
    print("다음: uv run python run.py  (서버 재시작 후 http://localhost:8000 확인)")


if __name__ == "__main__":
    main()
