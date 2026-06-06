"""
Grounding DINO + autodistill로 커스텀 이미지 자동 라벨링.
라벨링 완료 후 LabelImg를 클래스별로 순서대로 자동 실행해 FP 제거.

사용법:
    uv run python train/auto_label.py

전략:
  - CONF_THRESHOLD 낮게 설정 → FN(미탐) 최소화, FP(오탐)는 LabelImg에서 제거
  - LabelImg는 클래스 순서대로 자동 실행 → 닫으면 다음 클래스 자동 열림
  - 각 폴더에 classes.txt 자동 생성 → LabelImg가 YOLO 포맷 자동 인식

주의:
  - 첫 실행 시 Grounding DINO 모델 자동 다운로드 (~700MB)
  - Mac MPS / CPU에서도 동작
"""

import subprocess
import sys
from pathlib import Path

try:
    from autodistill.detection import CaptionOntology
    from autodistill_grounding_dino import GroundingDINO
except ImportError:
    subprocess.run(["uv", "add", "autodistill", "autodistill-grounding-dino"], check=True)
    from autodistill.detection import CaptionOntology
    from autodistill_grounding_dino import GroundingDINO

CUSTOM_ONTOLOGY = {
    "cola-custom":        "coca cola can red aluminum cylindrical",
    "water-custom":       "transparent plastic water bottle mineral water",
    "cereal-box-custom":  "cereal box rectangular cardboard breakfast",
    "paper-cup-custom":   "white paper cup disposable drinking cup",
    "cup-noodle-custom":  "cup noodle instant ramen cylindrical cup",
}

# FN 최소화를 위해 낮게 설정 — FP는 LabelImg 후처리에서 제거
CONF_THRESHOLD = 0.15
DATA_DIR = Path(__file__).parent / "data"


def ensure_labelimg():
    """labelimg 설치 확인 및 자동 설치."""
    try:
        subprocess.run(["labelimg", "--help"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[labelimg] 설치 중...")
        subprocess.run(["uv", "tool", "install", "labelimg"], check=True)
        print("[labelimg] 설치 완료")


def write_classes_txt(labels_dir: Path, cls_name: str):
    """LabelImg YOLO 포맷 자동 인식을 위한 classes.txt 생성."""
    # 클래스명에서 -custom 제거해서 표시
    display_name = cls_name.replace("-custom", "")
    (labels_dir / "classes.txt").write_text(display_name + "\n")


def label_class(cls_name: str, prompt: str):
    images_dir = DATA_DIR / cls_name / "train" / "images"
    labels_dir = DATA_DIR / cls_name / "train" / "labels"

    if not images_dir.exists():
        print(f"[{cls_name}] images/ 폴더 없음 — init_custom_dirs.py 먼저 실행하세요")
        return False

    imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not imgs:
        print(f"[{cls_name}] 이미지 없음, 건너뜀")
        return False

    labels_dir.mkdir(parents=True, exist_ok=True)
    write_classes_txt(labels_dir, cls_name)

    print(f"\n[{cls_name}] {len(imgs)}장 자동 라벨링 시작...")
    print(f"  프롬프트: \"{prompt}\"  |  threshold: {CONF_THRESHOLD}")

    ontology = CaptionOntology({prompt: cls_name})
    model = GroundingDINO(ontology=ontology, box_threshold=CONF_THRESHOLD)

    for ext in ["*.jpg", "*.png"]:
        if list(images_dir.glob(ext)):
            model.label(
                input_folder=str(images_dir),
                extension=ext.lstrip("*"),
                output_folder=str(labels_dir),
            )

    labeled = len([f for f in labels_dir.glob("*.txt") if f.name != "classes.txt"])
    print(f"[{cls_name}] 라벨 {labeled}/{len(imgs)}장 생성 완료")
    return True


def review_with_labelimg(cls_name: str):
    """LabelImg 실행 — 닫을 때까지 블로킹 (순서 보장)."""
    images_dir = DATA_DIR / cls_name / "train" / "images"
    labels_dir = DATA_DIR / cls_name / "train" / "labels"
    classes_txt = labels_dir / "classes.txt"

    print(f"\n[LabelImg] {cls_name} 검수 시작")
    print("  FP(오탐) bbox를 선택 후 Delete 키로 제거하세요.")
    print("  저장(Ctrl+S) 후 창을 닫으면 다음 클래스로 넘어갑니다.\n")

    subprocess.run([
        "labelimg",
        str(images_dir),
        str(classes_txt),
        str(labels_dir),
    ])


def main():
    print("=== Step 1 / 2 : Grounding DINO 자동 라벨링 ===")
    print(f"threshold={CONF_THRESHOLD}  (낮게 설정 → FP는 LabelImg에서 제거)\n")

    labeled_classes = []
    for cls_name, prompt in CUSTOM_ONTOLOGY.items():
        if label_class(cls_name, prompt):
            labeled_classes.append(cls_name)

    if not labeled_classes:
        print("\n처리할 이미지가 없습니다. init_custom_dirs.py → 이미지 복사 후 재실행")
        sys.exit(0)

    print(f"\n=== Step 2 / 2 : LabelImg FP 후처리 ({len(labeled_classes)}개 클래스) ===")
    ensure_labelimg()
    print("각 클래스별로 LabelImg가 순서대로 열립니다.")
    print("조작법:  Delete = bbox 삭제  |  Ctrl+S = 저장  |  창 닫기 = 다음 클래스\n")

    for cls_name in labeled_classes:
        review_with_labelimg(cls_name)

    print("\n=== 모든 클래스 검수 완료 ===")
    print("다음: uv run python train/augment.py")


if __name__ == "__main__":
    main()
