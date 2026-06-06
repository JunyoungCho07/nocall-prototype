"""촬영한 웹캠 이미지에 bounding box 라벨링.

목적:
    capture.py로 모은 이미지에 '제품 위치(박스)'를 표시해 YOLO 학습용 라벨을 만든다.
    클래스는 폴더별로 이미 정해져 있으므로, 박스만 드래그하면 된다 (LabelImg 설정 불필요).

사용법:
    uv run python train/label_tool.py

조작 (각 이미지마다):
    - 마우스로 제품을 감싸는 사각형을 드래그
    - ENTER 또는 SPACE : 그 박스 확정 (한 사진에 여러 제품이면 계속 드래그 후 각각 ENTER)
    - 박스 다 그렸으면 ESC : 다음 이미지로
    - 제품이 없거나 건너뛰려면 박스 없이 ESC
    - 창 닫기/중단: c 키

저장:
    train/data/webcam/labels/<class>/<name>.txt   (YOLO format: class_id xc yc w h, 정규화)
    이미 라벨 파일이 있는 이미지는 건너뜀(재실행해도 이어서 작업 가능).
"""
from pathlib import Path

import cv2

CLASSES = ["cola", "water", "snack-bag", "cereal-box", "cup-noodle"]
IMG = Path(__file__).parent / "data" / "webcam" / "images"
LBL = Path(__file__).parent / "data" / "webcam" / "labels"


def main():
    total_done = 0
    for cls_id, cls in enumerate(CLASSES):
        img_dir = IMG / cls
        if not img_dir.exists():
            continue
        lbl_dir = LBL / cls
        lbl_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(img_dir.glob("*.jpg"))
        todo = [p for p in images if not (lbl_dir / (p.stem + ".txt")).exists()]
        if not todo:
            print(f"[{cls}] 모두 라벨됨, 건너뜀")
            continue
        print(f"\n[{cls}] {len(todo)}장 라벨링 시작 "
              f"(드래그→ENTER, 다음:ESC, 중단:c)")

        for img_path in todo:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            win = f"{cls}  ({img_path.name})  drag box -> ENTER, done -> ESC"
            rois = cv2.selectROIs(win, img, showCrosshair=False, fromCenter=False)
            cv2.destroyAllWindows()

            lines = []
            for (x, y, bw, bh) in rois:
                if bw < 5 or bh < 5:
                    continue
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            (lbl_dir / (img_path.stem + ".txt")).write_text("\n".join(lines))
            total_done += 1

    print(f"\n=== 라벨링 완료: {total_done}장 ===")
    print("다음: uv run python train/add_webcam.py")


if __name__ == "__main__":
    main()
