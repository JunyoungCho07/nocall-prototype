"""학습된 모델 검증 — nocall.pt가 5개 클래스를 제대로 인식하는지 빠르게 확인.

사용법:
    uv run python train/verify.py                # 커스텀 원본(웹캠) 이미지로 검증
    uv run python train/verify.py <이미지_폴더>   # 지정 폴더로 검증

동작:
    - models/nocall.pt 로드 (없으면 git lfs pull 안내)
    - 각 이미지 추론 → 클래스별 검출 수 · 평균 신뢰도 집계
    - 박스 그린 결과를 runs/verify/ 에 저장 (눈으로 확인)
    - 합격 기준(클래스별 평균 신뢰도 ≥ 0.5) 자동 판정

웹캠 없이 모델 자체를 검증할 때 사용. 실제 동작 확인은 run.py(앱)로.
"""
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
MODEL = ROOT / "models" / "nocall.pt"
OUT_DIR = ROOT / "runs" / "verify"
CONF = 0.30
PASS_CONF = 0.50  # 클래스별 평균 신뢰도 합격선


def gather_default_images():
    """커스텀 원본(aug_ 제외) = 웹캠 도메인 이미지."""
    imgs = []
    for d in (Path(__file__).parent / "data").glob("*-custom/train/images"):
        for p in list(d.glob("*.jpg")) + list(d.glob("*.png")):
            if not p.stem.startswith("aug_"):
                imgs.append(p)
    return imgs


def main():
    if not MODEL.exists():
        print(f"모델 없음: {MODEL}")
        print("→ git lfs install && git lfs pull  로 모델을 받으세요.")
        sys.exit(1)

    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    else:
        imgs = gather_default_images()

    if not imgs:
        print("검증할 이미지가 없습니다.")
        sys.exit(1)

    model = YOLO(str(MODEL))
    names = model.names
    print(f"모델: {MODEL.name} | 클래스: {list(names.values())}")
    print(f"검증 이미지: {len(imgs)}장\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    confs = defaultdict(list)
    no_det = 0

    for p in imgs:
        res = model.predict(str(p), conf=CONF, verbose=False)[0]
        if len(res.boxes) == 0:
            no_det += 1
        for b in res.boxes:
            confs[int(b.cls)].append(float(b.conf))
        cv2.imwrite(str(OUT_DIR / f"verify_{p.stem}.jpg"), res.plot())

    print(f"{'클래스':<14}{'검출수':>6}{'평균신뢰도':>12}{'판정':>8}")
    print("-" * 42)
    all_ok = True
    for cid, name in names.items():
        cl = confs.get(cid, [])
        avg = sum(cl) / len(cl) if cl else 0.0
        ok = avg >= PASS_CONF and len(cl) > 0
        all_ok &= ok
        print(f"{name:<14}{len(cl):>6}{avg:>11.2f}{'  PASS' if ok else '  CHECK':>8}")

    print("-" * 42)
    print(f"검출 0개 이미지: {no_det}/{len(imgs)}장")
    print(f"시각 확인용 결과 이미지: {OUT_DIR}")
    print(f"\n=== 종합: {'PASS ✅' if all_ok else 'CHECK ⚠️ (낮은 클래스는 추가 촬영/재학습 권장)'} ===")
    print("실제 동작 검증: uv run python run.py → http://localhost:8000")


if __name__ == "__main__":
    main()
