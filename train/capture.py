"""웹캠으로 실제 제품 학습 이미지 수집.

목적:
    Roboflow 데이터(깔끔한 제품 사진)와 실제 매장 웹캠 환경의 '도메인 차이'를
    없애기 위해, 실제 사용할 카메라·조명·제품으로 직접 촬영한다.

사용법:
    uv run python train/capture.py

조작키 (영상 창에 포커스를 둔 상태에서):
    0 ~ 4  : 촬영할 클래스 선택  (0 cola / 1 water / 2 snack-bag / 3 cereal-box / 4 cup-noodle)
    SPACE  : 현재 화면 1장 촬영
    a      : 자동 연속 촬영 ON/OFF (0.4초마다 1장)
    q      : 종료

저장 위치:
    train/data/webcam/images/<class>/<class>_NNNN.jpg

촬영 팁 (정확도의 핵심):
    - 실제로 매대/카트에 쓸 '바로 그 제품'을 촬영할 것
    - 한 제품당 30~50장 이상, 각도·거리·회전·조명을 다양하게
    - 손에 들거나, 카트에 넣는 실제 동작 그대로도 촬영
    - 배경도 실제 환경과 같게 (특정 배경만 찍으면 배경을 외워버림)
"""
import time
from pathlib import Path

import cv2

CLASSES = ["cola", "water", "snack-bag", "cereal-box", "cup-noodle"]
OUT = Path(__file__).parent / "data" / "webcam" / "images"
AUTO_INTERVAL = 0.4  # 자동 촬영 간격(초)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("ERROR: 웹캠을 열 수 없습니다. 다른 앱(서버 포함)이 카메라를 쓰고 있지 않은지 확인하세요.")
        return

    counts = {}
    for c in CLASSES:
        d = OUT / c
        d.mkdir(parents=True, exist_ok=True)
        counts[c] = len(list(d.glob("*.jpg")))

    cls_idx = 0
    auto = False
    last = 0.0

    print("촬영 시작. 영상 창에서 0~4로 클래스 선택, SPACE 촬영, a 자동, q 종료")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        cls = CLASSES[cls_idx]

        disp = frame.copy()
        cv2.putText(disp, f"[{cls_idx}] {cls}   saved={counts[cls]}   auto={'ON' if auto else 'off'}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 0), 2)
        cv2.putText(disp, "0-4:class   SPACE:capture   a:auto   q:quit",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("NoCall capture", disp)

        k = cv2.waitKey(1) & 0xFF
        save = False
        if k == ord("q"):
            break
        elif k == ord("a"):
            auto = not auto
            last = time.time()
        elif ord("0") <= k <= ord("4"):
            cls_idx = k - ord("0")
        elif k == ord(" "):
            save = True

        if auto and (time.time() - last) > AUTO_INTERVAL:
            save = True

        if save:
            counts[cls] += 1
            fn = OUT / cls / f"{cls}_{counts[cls]:04d}.jpg"
            cv2.imwrite(str(fn), frame)
            last = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print("\n=== 촬영 완료 ===")
    for c in CLASSES:
        print(f"  {c:12s}: {counts[c]}장")
    print("다음: uv run python train/label_tool.py")


if __name__ == "__main__":
    main()
