import threading
from collections import defaultdict
import cv2
from ultralytics import YOLO
from pathlib import Path
from .cart import CartManager

MODEL_PATH = Path(__file__).parent.parent / "models" / "nocall.pt"
FALLBACK_MODEL = "yolov8n.pt"  # 학습 전 테스트용


class CameraProcessor:
    def __init__(self, cart: CartManager, camera_index: int = 0, conf: float = 0.30):
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

            # YOLO 일반 추론 — 추적/라인 없이, 이번 프레임에 '보이는' 상품을 센다
            results = self.model(frame, conf=self.conf, verbose=False)

            counts: dict[int, int] = defaultdict(int)
            boxes = results[0].boxes
            if boxes is not None and len(boxes):
                for box, cls, conf in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                ):
                    cid = int(cls)
                    counts[cid] += 1

                    # bounding box + 클래스명·신뢰도 표시
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 180), 2)
                    label = f"{self.model.names[cid]} {conf * 100:.0f}%"
                    cv2.putText(frame, label, (int(x1), int(y1) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 180), 2)

            self.cart.update_counts(counts)

            # 카트 현황 오버레이 (평활화 후 확정 수량)
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
