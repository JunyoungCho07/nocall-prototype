"""
Grounding DINO + autodistill로 커스텀 이미지 자동 라벨링.

사용법:
    uv run python train/auto_label.py

동작:
  - train/data/*-custom/train/images/ 의 이미지를 읽어
    Grounding DINO로 bbox 자동 감지
  - YOLO format (.txt) 라벨을 train/data/*-custom/train/labels/ 에 저장
  - merge.py가 이후 class 0 → 각 nocall class ID로 재매핑

주의:
  - 첫 실행 시 Grounding DINO 모델 자동 다운로드 (~700MB)
  - Mac MPS / CPU에서도 동작 (GPU보다 느리지만 이미지 수 적으므로 무관)
  - 신뢰도 낮은 결과는 CONF_THRESHOLD로 필터링
"""

from pathlib import Path

try:
    from autodistill.detection import CaptionOntology
    from autodistill_grounding_dino import GroundingDINO
except ImportError:
    import subprocess, sys
    subprocess.run(["uv", "add", "autodistill", "autodistill-grounding-dino"], check=True)
    from autodistill.detection import CaptionOntology
    from autodistill_grounding_dino import GroundingDINO

# 각 커스텀 폴더 → (텍스트 프롬프트, 폴더명)
# merge.py가 각 폴더 내 class ID를 nocall class로 재매핑하므로
# 여기서는 폴더당 단일 클래스 프롬프트만 사용 (출력 class_id = 0)
CUSTOM_ONTOLOGY = {
    "cola-custom":       "coca cola can red aluminum cylindrical",
    "water-custom":      "transparent plastic water bottle mineral water",
    "snack-bag-custom":  "potato chips snack bag foil package",
    "cereal-box-custom": "cereal box rectangular cardboard breakfast",
    "paper-cup-custom":  "white paper cup disposable drinking cup",
}

CONF_THRESHOLD = 0.25   # 낮을수록 더 많이 잡지만 오탐 증가
DATA_DIR = Path(__file__).parent / "data"


def label_class(cls_name: str, prompt: str):
    images_dir = DATA_DIR / cls_name / "train" / "images"
    labels_dir = DATA_DIR / cls_name / "train" / "labels"

    if not images_dir.exists():
        print(f"[{cls_name}] images/ 폴더 없음 — init_custom_dirs.py 먼저 실행하세요")
        return

    imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not imgs:
        print(f"[{cls_name}] 이미지 없음, 건너뜀")
        return

    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{cls_name}] {len(imgs)}장 자동 라벨링 시작...")
    print(f"  프롬프트: \"{prompt}\"")

    ontology = CaptionOntology({prompt: cls_name})
    model = GroundingDINO(ontology=ontology, box_threshold=CONF_THRESHOLD)

    # autodistill label() 호출 — images_dir의 이미지를 처리하고
    # 같은 폴더에 라벨 저장 후 labels_dir로 이동
    model.label(
        input_folder=str(images_dir),
        extension=".jpg",
        output_folder=str(labels_dir),
    )

    # .png 이미지도 처리
    png_imgs = list(images_dir.glob("*.png"))
    if png_imgs:
        model.label(
            input_folder=str(images_dir),
            extension=".png",
            output_folder=str(labels_dir),
        )

    labeled = len(list(labels_dir.glob("*.txt")))
    print(f"[{cls_name}] 완료 — {labeled}/{len(imgs)}장 라벨 생성")


def main():
    print("=== Grounding DINO 자동 라벨링 ===")
    print("첫 실행 시 모델 다운로드 (~700MB) 자동 진행\n")

    for cls_name, prompt in CUSTOM_ONTOLOGY.items():
        label_class(cls_name, prompt)

    print("\n=== 완료 ===")
    print("라벨 확인 후: uv run python train/merge.py")
    print("\n[선택] 라벨 검수가 필요하면:")
    print("  uv tool install labelimg")
    print("  labelimg train/data/<class>-custom/train/images train/data/<class>-custom/train/labels")


if __name__ == "__main__":
    main()
