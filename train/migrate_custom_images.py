"""
zip 파일에서 커스텀 이미지를 *-custom/train/images/ 로 마이그레이션.

사용법:
    uv run python train/migrate_custom_images.py
"""
import zipfile
import shutil
from pathlib import Path

ZIP_PATH = Path(__file__).parent.parent / "YOLO 학습 이미지-20260606T161127Z-3-001.zip"
DATA_DIR  = Path(__file__).parent / "data"

# zip 내 폴더명 → custom 폴더명 매핑
CLASS_MAP = {
    "cola":       "cola-custom",
    "water":      "water-custom",
    "cereal-box": "cereal-box-custom",
    "paper-cup":  "paper-cup-custom",
    "cup-noodle": "cup-noodle-custom",
}


def main():
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} 없음")
        return

    # 대상 폴더 생성
    for custom_name in CLASS_MAP.values():
        (DATA_DIR / custom_name / "train" / "images").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / custom_name / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / custom_name / "valid" / "images").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / custom_name / "valid" / "labels").mkdir(parents=True, exist_ok=True)

    counts = {k: 0 for k in CLASS_MAP}

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for info in zf.infolist():
            # 파일명 디코딩 (CP437 → UTF-8 fallback)
            try:
                name = info.filename.encode("cp437").decode("utf-8")
            except (UnicodeDecodeError, UnicodeEncodeError):
                name = info.filename

            # 이미지 파일만 처리
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # 클래스 폴더 판별
            parts = Path(name).parts
            cls = None
            for part in parts:
                if part in CLASS_MAP:
                    cls = part
                    break
            if cls is None:
                continue

            dest_dir = DATA_DIR / CLASS_MAP[cls] / "train" / "images"
            dest_file = dest_dir / Path(name).name

            with zf.open(info) as src, open(dest_file, "wb") as dst:
                shutil.copyfileobj(src, dst)
            counts[cls] += 1

    print("=== 마이그레이션 완료 ===")
    for cls, custom in CLASS_MAP.items():
        print(f"  {cls:12s} → {custom:20s}  {counts[cls]}장")
    print(f"\n총 {sum(counts.values())}장 복사 완료")
    print("다음: uv run python train/auto_label.py")


if __name__ == "__main__":
    main()
