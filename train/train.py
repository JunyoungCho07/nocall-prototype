"""YOLOv8 fine-tune on NoCall 5-class dataset.

Usage:
    uv run python train/train.py
"""
import shutil
import tempfile
import torch
from pathlib import Path
from ultralytics import YOLO

TRAIN_DIR = Path(__file__).parent
DATA_DIR  = TRAIN_DIR / "data"
EPOCHS    = 80
IMGSZ     = 640
BATCH     = 16  # RTX 4070 8GB 기준; OOM 시 8로 낮출 것


def _make_yaml() -> str:
    # OS에 무관하게 절대경로를 런타임에 주입한 임시 yaml 반환
    template = (TRAIN_DIR / "nocall.yaml").read_text()
    content  = template.replace("PLACEHOLDER", str(DATA_DIR))
    tmp      = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


def main():
    device = "cuda" if torch.cuda.is_available() else \
             "mps"  if torch.backends.mps.is_available() else "cpu"
    print(f"학습 디바이스: {device}")

    yaml_path = _make_yaml()
    model = YOLO("yolov8s.pt")
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        project="runs/train",
        name="nocall",
        exist_ok=True,
        patience=20,
        augment=True,
        degrees=10,
        scale=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    dest = TRAIN_DIR.parent / "models" / "nocall.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best, dest)
    print(f"모델 저장 완료: {dest}")


if __name__ == "__main__":
    main()
