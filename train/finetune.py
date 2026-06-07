"""2단계: 사전학습 모델을 custom(증강)+Roboflow서브샘플 세트로 미세조정.

전략 (방식 B):
    1단계 train.py가 만든 models/nocall.pt를 출발점으로,
    웹캠 도메인(custom)에 적응시킨다. 소량 custom에 과적합/망각을 막기 위해
    - backbone을 freeze (얕은 특징은 보존, 상위 레이어만 적응)
    - 낮은 학습률
    - Roboflow 서브샘플을 '닻'으로 함께 학습 (merge.py finetune이 구성)

사용법:
    1. uv run python train/merge.py finetune   # data/ft/ 구성
    2. uv run python train/finetune.py

GPU 자동 사용. 완료 시 models/nocall.pt 갱신 (= 최종 배포 모델).
"""
import shutil
import tempfile
from pathlib import Path

import torch
from ultralytics import YOLO

TRAIN_DIR = Path(__file__).parent
FT_DIR = TRAIN_DIR / "data" / "ft"        # merge.py finetune 출력 위치
BASE = TRAIN_DIR.parent / "models" / "nocall.pt"  # 1단계 사전학습 가중치

EPOCHS = 40
IMGSZ = 640
BATCH = 16          # OOM 시 8로
FREEZE = 10         # yolov8 backbone(0~9) 동결, head/neck만 학습
LR0 = 0.002         # 미세조정은 낮은 lr


def _make_ft_yaml() -> str:
    """nocall.yaml의 names/nc를 재사용하되 경로를 data/ft로 지정한 임시 yaml."""
    template = (TRAIN_DIR / "nocall.yaml").read_text()
    content = template.replace("PLACEHOLDER", str(FT_DIR))
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


def main():
    if not (FT_DIR / "images" / "train").exists():
        print("data/ft 가 없습니다. 먼저: uv run python train/merge.py finetune")
        return

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    start = str(BASE) if BASE.exists() else "yolov8s.pt"
    print(f"미세조정 시작 | device={device} | 출발 가중치={start} | freeze={FREEZE} | lr0={LR0}")

    yaml_path = _make_ft_yaml()
    model = YOLO(start)
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        freeze=FREEZE,
        lr0=LR0,
        project="runs/train",
        name="nocall_ft",
        exist_ok=True,
        patience=15,
        augment=True,
        degrees=10,
        scale=0.4,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    dest = TRAIN_DIR.parent / "models" / "nocall.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best, dest)
    print(f"최종 모델 저장 완료: {dest}")
    print("다음: uv run python run.py  (http://localhost:8000 확인)")


if __name__ == "__main__":
    main()
