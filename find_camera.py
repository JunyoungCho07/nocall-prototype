"""
연결된 카메라 번호를 찾는 도구.

    uv run python find_camera.py

각 번호(0~5)를 열어보고, 영상이 잡히는 번호를 출력합니다.
- macOS 내장 카메라는 보통 0번
- 외부 USB 웹캠은 보통 1번(가끔 2번)
잡히는 번호를 .env의 CAMERA_INDEX 에 적으면 됩니다.
"""
import cv2

print("카메라 탐색 중...\n")
found = []
for idx in range(6):
    cap = cv2.VideoCapture(idx)
    ok, frame = cap.read()
    if ok and frame is not None:
        h, w = frame.shape[:2]
        print(f"  [{idx}] 사용 가능  ({w}x{h})")
        found.append(idx)
    cap.release()

print()
if not found:
    print("잡히는 카메라가 없습니다. 웹캠 연결 / 카메라 권한을 확인하세요.")
else:
    print(f"사용 가능한 번호: {found}")
    print("외부 웹캠은 보통 가장 큰 번호입니다.")
    print("원하는 번호를 .env 파일에 적으세요:  CAMERA_INDEX=<번호>")
