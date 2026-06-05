import threading
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from .cart import CartManager

MODEL_PATH = Path(__file__).parent.parent / "models" / "nocall.pt"
FALLBACK_MODEL = "yolov8n.pt"  # 학습 전 테스트용


class CameraProcessor:
    def __init__(self, cart: CartManager, camera_index: int = 0, conf: float = 0.45):
        self.cart = cart
        self.camera_index = camera_index
        self.conf = conf
        self._frame: bytes | None = None
        self._lock = threading.Lock()
        self._running = False

        model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL
        self.model = YOLO(str(model_path))

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self._running:
            ok, frame = cap.read()
            if not ok:
                continue

            frame_h, frame_w = frame.shape[:2]
            line_y = int(frame_h * self.cart.line_y_ratio)

            # YOLO + ByteTrack
            results = self.model.track(
                frame, persist=True, conf=self.conf, tracker="bytetrack.yaml", verbose=False
            )

            tracks = []
            if results[0].boxes.id is not None:
                for box, cls, tid in zip(
                    results[0].boxes.xyxy.cpu().numpy(),
                    results[0].boxes.cls.cpu().numpy(),
                    results[0].boxes.id.cpu().numpy(),
                ):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    tracks.append((int(tid), int(cls), cx, cy))

                    # bounding box 그리기
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 180), 2)
                    label = f"ID{int(tid)}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 180), 2)

            self.cart.process_tracks(tracks, frame_h)

            # 가상 라인 시각화
            cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 80, 255), 2)
            cv2.putText(frame, "SCAN LINE", (10, line_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)

            # 카트 현황 오버레이
            items = self.cart.get_cart()
            overlay_lines = [f"{i['name']} x{i['qty']}" for i in items]
            for idx, text in enumerate(overlay_lines):
                cv2.putText(frame, text, (10, 30 + idx * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                self._frame = buf.tobytes()

        cap.release()

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._frame
