"""
Roboflow Universe에서 5개 제품 데이터셋 다운로드.

사용법:
    1. .env 파일에 ROBOFLOW_API_KEY 입력
    2. uv run python train/download.py

다운로드 위치: train/data/ (nocall.yaml과 동일 경로)
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY or API_KEY == "your_api_key_here":
    print("ERROR: .env 파일에 ROBOFLOW_API_KEY를 입력하세요.")
    sys.exit(1)

try:
    from roboflow import Roboflow
except ImportError:
    import subprocess
    subprocess.run(["uv", "add", "roboflow"], check=True)
    from roboflow import Roboflow

# Roboflow Universe 데이터셋 목록
# universe.roboflow.com에서 각 데이터셋 Download → YOLOv8 → "show download code" 참고
DATASETS = [
    # (workspace, project, version, local_name)
    ("proba-yoloa",    "coca-cola-can-detection-n8az7", 1, "cola"),
    ("yolo-nznfs",     "plastic-bottles-ip5yb-uziag",   1, "water"),
    ("robocup2022-kogzd", "potatochip",       1, "snack-bag"),
    ("deep-learners",    "cereal-box-dqeyy", 1, "cereal-box"),
    ("mmmmmm",           "ramen-iqwqm",      5, "cup-noodle"),
]

DATA_DIR = Path(__file__).parent / "data"


def main():
    rf = Roboflow(api_key=API_KEY)

    for workspace, project_name, version_num, local_name in DATASETS:
        dest = DATA_DIR / local_name
        # 폴더 안에 실제 이미지가 있을 때만 건너뜀 (빈 폴더는 다시 다운로드)
        has_images = dest.exists() and (
            any(dest.rglob("*.jpg")) or any(dest.rglob("*.png"))
        )
        if has_images:
            print(f"[{local_name}] 이미 존재, 건너뜀 → {dest}")
            continue

        print(f"\n[{local_name}] 다운로드 중... ({workspace}/{project_name} v{version_num})")
        try:
            project = rf.workspace(workspace).project(project_name)
            version = project.version(version_num)
            version.download("yolov8", location=str(dest))
            print(f"[{local_name}] 완료 → {dest}")
        except Exception as e:
            print(f"[{local_name}] 실패: {e}")
            print(f"  → universe.roboflow.com 에서 직접 확인 후 workspace/project/version 수정 필요")

    print("\n=== 다운로드 완료 ===")
    print("다음: uv run python train/merge.py  (데이터 통합)")


if __name__ == "__main__":
    main()
