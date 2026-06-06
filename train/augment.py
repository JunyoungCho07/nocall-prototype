"""
커스텀 데이터 오프라인 증강 스크립트.

목적:
    직접 촬영한 이미지(20~30장)를 albumentations로 증강 → ~200장으로 확대.
    Roboflow 데이터와 합산해 클래스 불균형 완화.

사용법:
    uv run python train/augment.py

실행 순서:
    init_custom_dirs.py → (촬영 + auto_label.py) → augment.py → merge.py → train.py

동작:
    - train/data/*-custom/train/images/ + labels/ 를 읽어
    - AUG_PER_IMAGE 배수만큼 증강 이미지 + 라벨을 같은 폴더에 추가
    - 원본 파일은 유지 (aug_ prefix로 신규 생성)
"""

from pathlib import Path
import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:
    import subprocess
    subprocess.run(["uv", "add", "albumentations", "opencv-python-headless"], check=True)
    import albumentations as A

AUG_PER_IMAGE = 7   # 원본 1장 → 증강 7장 추가 (총 8배)

CUSTOM_CLASSES = [
    "cola-custom",
    "water-custom",
    "cereal-box-custom",
    "paper-cup-custom",
    "cup-noodle-custom",
]

DATA_DIR = Path(__file__).parent / "data"

# 카트 -45도 환경에 맞춘 증강 파이프라인
TRANSFORM = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Rotate(limit=15, p=0.7),
        A.Perspective(scale=(0.02, 0.08), p=0.4),
        A.HorizontalFlip(p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.Cutout(num_holes=2, max_h_size=20, max_w_size=20, p=0.3),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,   # bbox가 30% 이상 보여야 유지
    ),
)


def load_yolo_label(label_path: Path):
    """YOLO txt → [(class_id, cx, cy, w, h), ...]"""
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4])))
    return rows


def save_yolo_label(label_path: Path, boxes, class_labels):
    lines = []
    for cls, (cx, cy, w, h) in zip(class_labels, boxes):
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines))


def augment_class(cls_name: str):
    img_dir = DATA_DIR / cls_name / "train" / "images"
    lbl_dir = DATA_DIR / cls_name / "train" / "labels"

    if not img_dir.exists():
        print(f"[{cls_name}] 폴더 없음, 건너뜀")
        return

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    # 이미 증강된 파일은 제외 (aug_ prefix)
    originals = [p for p in images if not p.stem.startswith("aug_")]

    if not originals:
        print(f"[{cls_name}] 원본 이미지 없음, 건너뜀")
        return

    print(f"\n[{cls_name}] 원본 {len(originals)}장 → ×{AUG_PER_IMAGE+1} 증강 시작...")
    generated = 0

    for img_path in originals:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        rows = load_yolo_label(lbl_path)

        bboxes = [(r[1], r[2], r[3], r[4]) for r in rows]
        class_labels = [r[0] for r in rows]

        for i in range(AUG_PER_IMAGE):
            try:
                result = TRANSFORM(
                    image=img_rgb,
                    bboxes=bboxes if bboxes else [],
                    class_labels=class_labels if class_labels else [],
                )
            except Exception:
                continue

            aug_name = f"aug_{img_path.stem}_{i:02d}"
            out_img = img_dir / f"{aug_name}.jpg"
            out_lbl = lbl_dir / f"{aug_name}.txt"

            aug_bgr = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_img), aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            save_yolo_label(out_lbl, result["bboxes"], result["class_labels"])
            generated += 1

    total = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
    print(f"[{cls_name}] 완료 — 원본 {len(originals)}장 + 증강 {generated}장 = 총 {total}장")


def main():
    print("=== 커스텀 데이터 오프라인 증강 ===")
    for cls in CUSTOM_CLASSES:
        augment_class(cls)
    print("\n=== 전처리 완료 ===")
    print("다음: uv run python train/merge.py")


if __name__ == "__main__":
    main()
