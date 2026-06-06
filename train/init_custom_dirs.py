"""
직접 촬영한 커스텀 이미지용 폴더 구조 생성.

사용법:
    uv run python train/init_custom_dirs.py

생성 후 워크플로우:
    1. 촬영한 이미지를 train/data/<class>-custom/train/images/ 에 복사
    2. Roboflow 웹 UI 또는 LabelImg로 라벨링
    3. 라벨 .txt 파일을 train/data/<class>-custom/train/labels/ 에 배치
    4. uv run python train/merge.py  (자동으로 class ID 매핑 후 통합)
"""
from pathlib import Path

CUSTOM_CLASSES = [
    "cola-custom",
    "water-custom",
    "cereal-box-custom",
    "paper-cup-custom",
    "cup-noodle-custom",
]

DATA_DIR = Path(__file__).parent / "data"


def main():
    for cls in CUSTOM_CLASSES:
        for split in ["train", "valid"]:
            for sub in ["images", "labels"]:
                d = DATA_DIR / cls / split / sub
                d.mkdir(parents=True, exist_ok=True)
        print(f"[{cls}] 폴더 생성 완료")
        print(f"  이미지 → train/data/{cls}/train/images/")
        print(f"  라벨   → train/data/{cls}/train/labels/")

    print("\n=== 완료 ===")
    print("촬영 이미지를 각 images/ 폴더에 넣은 뒤 라벨링하세요.")
    print("라벨링 후: uv run python train/merge.py")


if __name__ == "__main__":
    main()
