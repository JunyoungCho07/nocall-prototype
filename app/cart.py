import sqlite3
import threading
import uuid
import json
from collections import deque
from statistics import median
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
    """
    계산대 위에 '한 겹'으로 펼쳐 놓은 상품을 카메라가 전체로 비추는 방식.
    매 프레임 감지된 {class_id: count}를 받아, 최근 N프레임의 중앙값으로
    안정화한 개수를 장바구니로 확정한다 (프레임별 떨림 방지).
    """

    def __init__(self, window: int = 15):
        self._lock = threading.Lock()
        # 최근 N프레임의 {class_id: count} 스냅샷
        self._frames: deque[dict[int, int]] = deque(maxlen=window)
        # 평활화로 확정된 장바구니 {class_id: qty}
        self._cart: dict[int, int] = {}

    def update_counts(self, counts: dict[int, int]):
        """매 프레임 호출. counts = {class_id: 이번 프레임에 보인 개수}"""
        with self._lock:
            self._frames.append(dict(counts))
            self._recompute_locked()

    def _recompute_locked(self):
        """self._frames 중앙값으로 _cart 재계산 (반드시 lock 안에서 호출)."""
        if not self._frames:
            self._cart = {}
            return
        all_classes: set[int] = set()
        for f in self._frames:
            all_classes.update(f)
        stable: dict[int, int] = {}
        for cid in all_classes:
            vals = [f.get(cid, 0) for f in self._frames]
            qty = int(median(vals))
            if qty > 0:
                stable[cid] = qty
        self._cart = stable

    def get_cart(self) -> list[dict]:
        with self._lock:
            items = []
            for class_id, qty in self._cart.items():
                if qty > 0 and class_id in PRODUCTS:
                    p = PRODUCTS[class_id]
                    items.append({
                        "class_id": class_id,
                        "name":     p["name"],
                        "qty":      qty,
                        "price":    p["price"],
                        "subtotal": p["price"] * qty,
                    })
            items.sort(key=lambda i: i["class_id"])
            return items

    def get_total(self) -> int:
        return sum(i["subtotal"] for i in self.get_cart())

    def reset(self):
        with self._lock:
            self._frames.clear()
            self._cart = {}

    def save_receipt(self) -> str:
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
        con = sqlite3.connect(DB_PATH)
        row = con.execute(
            "SELECT items_json FROM receipts WHERE id = ?", (session_id,)
        ).fetchone()
        con.close()
        if row is None:
            return None
        return json.loads(row[0])
