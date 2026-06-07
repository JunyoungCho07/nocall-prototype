"""개별 데이터셋을 nocall 학습 포맷(class 0-4)으로 통합 — 2단계 학습 지원.

모드:
    uv run python train/merge.py pretrain   # 1단계: Roboflow만 → data/images/{train,val}
    uv run python train/merge.py finetune   # 2단계: custom(증강) + Roboflow 서브샘플 → data/ft/images/{train,val}
    (인자 없으면 pretrain)

전략 (방식 B — 혼합 finetune):
    - 1단계: Roboflow 전체로 일반 객체 표현 학습 (paper-cup은 과대 → cap)
    - 2단계: 직접 촬영(custom, 증강 포함)으로 도메인 적응 + Roboflow 일부를
             '닻'으로 섞어 치명적 망각(catastrophic forgetting) 방지
"""
import random
import shutil
import sys
from pathlib import Path

# download.py의 local_name 순서 = nocall class ID
ROBOFLOW = {
    "cola":       0,
    "water":      1,
    "cereal-box": 2,
    "paper-cup":  3,
    "cup-noodle": 4,
}
CUSTOM = {
    "cola-custom":       0,
    "water-custom":      1,
    "cereal-box-custom": 2,
    "paper-cup-custom":  3,
    "cup-noodle-custom": 4,
}

DATA_DIR = Path(__file__).parent / "data"
VAL_RATIO = 0.15

# 1단계 다운샘플링: paper-cup(Roboflow 3,181) 과대 → cereal-box(315)와의 격차 완화
PRETRAIN_CAP = {"paper-cup": 800}
# 2단계 Roboflow 서브샘플: 망각 방지용 '닻' (클래스당 N장)
FT_ROBOFLOW_PER_CLASS = 80


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


def collect_images(src_root: Path):
    """Roboflow(train/valid/test) · custom(train) 구조 모두에서 이미지 수집."""
    imgs = []
    for split in ["train", "valid", "test"]:
        d = src_root / split / "images"
        if d.exists():
            imgs += list(d.glob("*.jpg")) + list(d.glob("*.png"))
    return imgs


def make_out_dirs(root: Path):
    d = {
        "img": {"train": root / "images" / "train", "val": root / "images" / "val"},
        "lbl": {"train": root / "labels" / "train", "val": root / "labels" / "val"},
    }
    for grp in d.values():
        for p in grp.values():
            p.mkdir(parents=True, exist_ok=True)
            # 재실행 시 깨끗하게 — 기존 파일 제거
            for f in p.glob("*"):
                if f.is_file():
                    f.unlink()
    return d


def process_source(local_name: str, class_id: int, cap, d) -> int:
    src_root = DATA_DIR / local_name
    if not src_root.exists():
        print(f"[{local_name}] 폴더 없음, 건너뜀")
        return 0

    imgs = collect_images(src_root)
    if not imgs:
        print(f"[{local_name}] 이미지 없음")
        return 0

    random.shuffle(imgs)
    if cap and len(imgs) > cap:
        print(f"[{local_name}] 샘플링: {len(imgs)} → {cap}장")
        imgs = imgs[:cap]

    val_cut = max(1, int(len(imgs) * VAL_RATIO))
    splits = {"val": imgs[:val_cut], "train": imgs[val_cut:]}

    for split_name, split_imgs in splits.items():
        img_out = d["img"][split_name]
        lbl_out = d["lbl"][split_name]
        for img_path in split_imgs:
            new_name = f"{local_name}_{img_path.name}"
            shutil.copy(img_path, img_out / new_name)
            # 라벨 (구조: <split>/images/x.jpg ↔ <split>/labels/x.txt)
            lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            if lbl_path.exists():
                remap_label(lbl_path, lbl_out / f"{local_name}_{img_path.stem}.txt", class_id)

    print(f"[{local_name}] class {class_id} → {len(imgs)}장")
    return len(imgs)


def merge(mode: str):
    if mode == "pretrain":
        root = DATA_DIR
        d = make_out_dirs(root)
        print("=== 1단계 사전학습용 통합 (Roboflow only) ===")
        total = sum(process_source(n, c, PRETRAIN_CAP.get(n), d)
                    for n, c in ROBOFLOW.items())
    elif mode == "finetune":
        root = DATA_DIR / "ft"
        d = make_out_dirs(root)
        print("=== 2단계 미세조정용 통합 (custom 증강 + Roboflow 서브샘플) ===")
        total = 0
        for n, c in CUSTOM.items():            # custom 전체 (cap 없음)
            total += process_source(n, c, None, d)
        for n, c in ROBOFLOW.items():          # roboflow 닻 (클래스당 N장)
            total += process_source(n, c, FT_ROBOFLOW_PER_CLASS, d)
    else:
        print(f"알 수 없는 모드: {mode}  (pretrain | finetune)")
        sys.exit(1)

    print(f"\n총 {total}장 통합 완료")
    print(f"train: {len(list(d['img']['train'].glob('*')))}장 / "
          f"val: {len(list(d['img']['val'].glob('*')))}장")
    if mode == "pretrain":
        print("다음: uv run python train/train.py")
    else:
        print("다음: uv run python train/finetune.py")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pretrain"
    merge(mode)
