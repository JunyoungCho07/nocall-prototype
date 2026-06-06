"""
개별 다운로드된 데이터셋을 nocall.yaml 포맷(class 0-4)으로 통합.

사용법:
    uv run python train/merge.py

동작:
  - train/data/<local_name>/ 의 이미지·라벨을
    train/data/images/train|val / train/data/labels/train|val 로 병합
  - 각 데이터셋의 class 0 → nocall class ID로 재매핑
"""
import shutil
import random
from pathlib import Path

# download.py의 local_name 순서 = nocall class ID
CLASS_MAP = {
    "cola":             0,
    "water":            1,
    "snack-bag":        2,
    "cereal-box":       3,
    "paper-cup":        4,
    # 직접 촬영분 — 각 Roboflow 데이터셋과 동일 class로 병합
    "cola-custom":      0,
    "water-custom":     1,
    "snack-bag-custom": 2,
    "cereal-box-custom":3,
    "paper-cup-custom": 4,
}

DATA_DIR  = Path(__file__).parent / "data"
IMG_TRAIN = DATA_DIR / "images" / "train"
IMG_VAL   = DATA_DIR / "images" / "val"
LBL_TRAIN = DATA_DIR / "labels" / "train"
LBL_VAL   = DATA_DIR / "labels" / "val"

for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    d.mkdir(parents=True, exist_ok=True)

VAL_RATIO = 0.15


def remap_label(src: Path, dst: Path, new_class_id: int):
    lines = src.read_text().strip().splitlines()
    remapped = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        parts[0] = str(new_class_id)
        remapped.append(" ".join(parts))
    dst.write_text("\n".join(remapped))


def merge():
    total = 0
    for local_name, class_id in CLASS_MAP.items():
        src_root = DATA_DIR / local_name
        if not src_root.exists():
            print(f"[{local_name}] 폴더 없음, 건너뜀 (download.py 먼저 실행)")
            continue

        # Roboflow 다운로드 구조: train/images, valid/images, test/images
        imgs = []
        for split in ["train", "valid", "test"]:
            imgs += list((src_root / split / "images").glob("*.jpg"))
            imgs += list((src_root / split / "images").glob("*.png"))

        if not imgs:
            print(f"[{local_name}] 이미지 없음")
            continue

        random.shuffle(imgs)
        val_cut = max(1, int(len(imgs) * VAL_RATIO))
        splits = {"val": imgs[:val_cut], "train": imgs[val_cut:]}

        for split_name, split_imgs in splits.items():
            img_out = IMG_VAL   if split_name == "val" else IMG_TRAIN
            lbl_out = LBL_VAL   if split_name == "val" else LBL_TRAIN

            for img_path in split_imgs:
                # 이미지 복사 (클래스명 prefix로 파일명 충돌 방지)
                new_name = f"{local_name}_{img_path.name}"
                shutil.copy(img_path, img_out / new_name)

                # 라벨 재매핑 (구조: <split>/images/x.jpg ↔ <split>/labels/x.txt)
                lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
                if lbl_path.exists():
                    remap_label(lbl_path, lbl_out / f"{local_name}_{img_path.stem}.txt", class_id)

            total += len(split_imgs)
        print(f"[{local_name}] class {class_id} → {len(imgs)}장 병합 완료")

    print(f"\n총 {total}장 통합 완료")
    print(f"train: {len(list(IMG_TRAIN.glob('*')))}장 / val: {len(list(IMG_VAL.glob('*')))}장")
    print("다음: uv run python train/train.py")


if __name__ == "__main__":
    merge()
