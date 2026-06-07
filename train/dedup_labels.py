"""커스텀 라벨의 중복 bbox를 NMS로 제거.

목적:
    Grounding DINO가 낮은 threshold(0.15)로 같은 물체를 여러 번 검출해
    한 사진에 거의 동일한 박스가 2~4개씩 생긴다. IoU 기반 그리디 NMS로
    겹치는 박스를 하나로 합쳐 학습용 라벨을 정리한다.

사용법:
    uv run python train/dedup_labels.py

동작:
    - train/data/*-custom/train/labels/*.txt 의 박스를 IoU>THRESHOLD면 중복으로 보고 제거
    - YOLO 라벨엔 신뢰도가 없으므로 그리디 방식(먼저 본 박스 유지)
    - 서로 떨어진 박스(IoU 낮음)는 그대로 보존 → 사진에 제품이 2개면 둘 다 유지
"""
from pathlib import Path

CLASSES = ["cola-custom", "water-custom", "cereal-box-custom",
           "paper-cup-custom", "cup-noodle-custom"]
DATA = Path(__file__).parent / "data"
IOU_THRESHOLD = 0.5  # 이 이상 겹치면 같은 물체로 보고 하나만 남김


def to_xyxy(box):
    _, xc, yc, w, h = box
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def dedup_file(path: Path) -> tuple[int, int]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    boxes = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        boxes.append([parts[0]] + [float(x) for x in parts[1:5]])

    kept = []
    kept_xyxy = []
    for box in boxes:
        xyxy = to_xyxy(box)
        if all(iou(xyxy, k) <= IOU_THRESHOLD for k in kept_xyxy):
            kept.append(box)
            kept_xyxy.append(xyxy)

    out = "\n".join(
        f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in kept
    )
    path.write_text(out)
    return len(boxes), len(kept)


def main():
    before_total = after_total = files = 0
    for cls in CLASSES:
        ldir = DATA / cls / "train" / "labels"
        if not ldir.exists():
            continue
        for txt in ldir.glob("*.txt"):
            if txt.name == "classes.txt":
                continue
            b, a = dedup_file(txt)
            before_total += b
            after_total += a
            files += 1

    print(f"파일 {files}개 처리")
    print(f"박스: {before_total} → {after_total} "
          f"(평균 {before_total/files:.2f} → {after_total/files:.2f} 개/장)" if files else "처리할 파일 없음")
    print("다음: uv run python train/augment.py")


if __name__ == "__main__":
    main()
