import sqlite3
import threading
import uuid
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "nocall.db"

PRODUCTS = {
    0: {"name": "콜라",        "price": 1800},
    1: {"name": "물",          "price": 1000},
    2: {"name": "시리얼(박스)", "price": 6000},
    3: {"name": "종이컵",       "price":  500},
    4: {"name": "컵라면",       "price": 1500},
}


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            id TEXT PRIMARY KEY,
            created_at TEXT DEFAULT (datetime('now','localtime')),
            items_json TEXT
        )
    """)
    con.commit()
    con.close()


class CartManager:
    def __init__(self, line_y_ratio: float = 0.5):
        self._lock = threading.Lock()
        # {class_id: quantity}
        self._cart: dict[int, int] = defaultdict(int)
        # {track_id: last_y}  — ByteTrack ID → 직전 프레임 Y 중심
        self._track_prev_y: dict[int, float] = {}
        # track_id → 이미 라인 교차 처리된 ID (방향별)
        self._crossed: set[tuple[int, str]] = set()
        self.line_y_ratio = line_y_ratio  # 프레임 높이 대비 라인 위치

    def process_tracks(self, tracks, frame_h: int):
        """
        tracks: list of (track_id, class_id, cx, cy)
        cy가 line_y를 위→아래 교차 → add
        아래→위 교차 → remove
        """
        line_y = frame_h * self.line_y_ratio
        current_ids = set()

        for track_id, class_id, _cx, cy in tracks:
            current_ids.add(track_id)
            prev_y = self._track_prev_y.get(track_id)

            if prev_y is not None:
                crossed_down = prev_y < line_y <= cy
                crossed_up   = prev_y > line_y >= cy

                if crossed_down and (track_id, "down") not in self._crossed:
                    self._crossed.add((track_id, "down"))
                    with self._lock:
                        self._cart[class_id] += 1

                elif crossed_up and (track_id, "up") not in self._crossed:
                    self._crossed.add((track_id, "up"))
                    with self._lock:
                        qty = self._cart.get(class_id, 0)
                        if qty > 0:
                            self._cart[class_id] = qty - 1

            self._track_prev_y[track_id] = cy

        # 사라진 track_id 정리
        gone = set(self._track_prev_y) - current_ids
        for tid in gone:
            del self._track_prev_y[tid]
            self._crossed.discard((tid, "down"))
            self._crossed.discard((tid, "up"))

    def get_cart(self) -> list[dict]:
        with self._lock:
            items = []
            for class_id, qty in self._cart.items():
                if qty > 0:
                    p = PRODUCTS[class_id]
                    items.append({
                        "class_id": class_id,
                        "name":     p["name"],
                        "qty":      qty,
                        "price":    p["price"],
                        "subtotal": p["price"] * qty,
                    })
            return items

    def get_total(self) -> int:
        return sum(i["subtotal"] for i in self.get_cart())

    def reset(self):
        with self._lock:
            self._cart.clear()
            self._track_prev_y.clear()
            self._crossed.clear()

    def save_receipt(self) -> str:
        import json
        session_id = uuid.uuid4().hex[:8]
        items = self.get_cart()
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO receipts (id, items_json) VALUES (?, ?)",
            (session_id, json.dumps(items, ensure_ascii=False)),
        )
        con.commit()
        con.close()
        return session_id

    def load_receipt(self, session_id: str) -> list[dict] | None:
        import json
        con = sqlite3.connect(DB_PATH)
        row = con.execute(
            "SELECT items_json FROM receipts WHERE id = ?", (session_id,)
        ).fetchone()
        con.close()
        if row is None:
            return None
        return json.loads(row[0])
