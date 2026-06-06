"""촬영·라벨링한 웹캠 데이터를 학습 세트에 합치기.

목적:
    train/data/webcam/ 의 이미지·라벨을 기존 학습 폴더
    (images/train|val, labels/train|val)에 추가한다.
    기존 Roboflow 데이터와 '섞어서' 학습하면 도메인 차이를 줄이면서
    기존 일반화 성능도 유지된다.

사용법:
    uv run python train/add_webcam.py

동작:
    - webcam/images/<class>/*.jpg + 짝 라벨을 train/val로 85:15 분할 복사
    - 파일명에 'webcam_' prefix → 기존 파일과 충돌 방지, 출처 구분 용이
    - 라벨이 없는(빈) 이미지는 건너뜀 (배경 전용으로 쓰려면 빈 라벨 허용하도록 수정)
"""
import random
import shutil
from pathlib import Path

CLASSES = ["cola", "water", "snack-bag", "cereal-box", "cup-noodle"]
DATA = Path(__file__).parent / "data"
WEBCAM_IMG = DATA / "webcam" / "images"
WEBCAM_LBL = DATA / "webcam" / "labels"
IMG_TRAIN = DATA / "images" / "train"
IMG_VAL = DATA / "images" / "val"
LBL_TRAIN = DATA / "labels" / "train"
LBL_VAL = DATA / "labels" / "val"
VAL_RATIO = 0.15

for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    d.mkdir(parents=True, exist_ok=True)


def main():
    total = 0
    for cls in CLASSES:
        img_dir = WEBCAM_IMG / cls
        lbl_dir = WEBCAM_LBL / cls
        if not img_dir.exists():
            continue

        pairs = []
        for img in img_dir.glob("*.jpg"):
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists() and lbl.read_text().strip():  # 라벨 있고 비어있지 않음
                pairs.append((img, lbl))

        if not pairs:
            print(f"[{cls}] 라벨된 이미지 없음, 건너뜀")
            continue

        random.shuffle(pairs)
        cut = max(1, int(len(pairs) * VAL_RATIO))
        splits = {"val": pairs[:cut], "train": pairs[cut:]}

        for split, items in splits.items():
            img_out = IMG_VAL if split == "val" else IMG_TRAIN
            lbl_out = LBL_VAL if split == "val" else LBL_TRAIN
            for img, lbl in items:
                shutil.copy(img, img_out / f"webcam_{img.name}")
                shutil.copy(lbl, lbl_out / f"webcam_{lbl.name}")
            total += len(items)
        print(f"[{cls}] {len(pairs)}장 추가 (train {len(splits['train'])} / val {len(splits['val'])})")

    print(f"\n총 {total}장 학습 세트에 추가 완료")
    print(f"현재 train: {len(list(IMG_TRAIN.glob('*')))}장 / val: {len(list(IMG_VAL.glob('*')))}장")
    print("다음: uv run python train/finetune.py")


if __name__ == "__main__":
    main()
