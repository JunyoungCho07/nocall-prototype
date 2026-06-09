import asyncio
import io
import os
import socket
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pathlib import Path
import qrcode
from dotenv import load_dotenv

from .cart import CartManager, init_db
from .camera import CameraProcessor

load_dotenv()

app = FastAPI(title="NoCall Prototype")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# .env 의 CAMERA_INDEX 로 카메라 선택 (없으면 0번=내장). 외부 웹캠은 보통 1번.
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

cart = CartManager()
cam = CameraProcessor(cart=cart, camera_index=CAMERA_INDEX)


@app.on_event("startup")
async def startup():
    init_db()
    cam.start()


@app.on_event("shutdown")
async def shutdown():
    cam.stop()


# ── 실시간 영상 스트림 ──────────────────────────────────────────
async def _frame_generator():
    while True:
        frame = cam.get_frame()
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        await asyncio.sleep(0.033)  # ~30fps


@app.get("/stream")
async def stream():
    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── 카트 상태 API ───────────────────────────────────────────────
@app.get("/cart")
async def get_cart():
    return {"items": cart.get_cart(), "total": cart.get_total()}


@app.post("/cart/reset")
async def reset_cart():
    cart.reset()
    return {"ok": True}


# ── 체크아웃 & QR 생성 ──────────────────────────────────────────
def _local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


@app.post("/checkout")
async def checkout():
    items = cart.get_cart()
    if not items:
        raise HTTPException(status_code=400, detail="카트가 비어있습니다")

    session_id = cart.save_receipt()
    ip = _local_ip()
    url = f"http://{ip}:8000/receipt/{session_id}"

    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="image/png",
        headers={"X-Receipt-URL": url, "X-Session-Id": session_id},
    )


# ── 영수증 페이지 ───────────────────────────────────────────────
@app.get("/receipt/{session_id}", response_class=HTMLResponse)
async def receipt(request: Request, session_id: str):
    items = cart.load_receipt(session_id)
    if items is None:
        raise HTTPException(status_code=404, detail="영수증을 찾을 수 없습니다")
    total = sum(i["subtotal"] for i in items)
    return templates.TemplateResponse(
        request,
        "receipt.html",
        {"items": items, "total": total, "session_id": session_id},
    )


# ── 데모용 메인 페이지 ──────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>NoCall — Smart Cart</title>
  <style>
    body { font-family: 'Pretendard', sans-serif; background:#1A1D2E; color:#fff; margin:0; display:flex; gap:24px; padding:24px; }
    h1 { color:#00B4A6; margin:0 0 16px; }
    #video-col { flex:2; }
    #cart-col { flex:1; min-width:260px; }
    img#stream { width:100%; border-radius:12px; }
    #cart-box { background:#23263a; border-radius:12px; padding:20px; }
    table { width:100%; border-collapse:collapse; }
    td,th { padding:8px 4px; border-bottom:1px solid #333; }
    th { color:#00B4A6; font-size:0.85rem; }
    #total-row td { font-weight:700; font-size:1.1rem; color:#00B4A6; border:none; padding-top:14px; }
    .btn { display:block; width:100%; margin-top:16px; padding:14px; border-radius:8px; border:none; font-size:1rem; cursor:pointer; }
    #btn-checkout { background:#00B4A6; color:#fff; }
    #btn-reset    { background:#444; color:#ccc; margin-top:8px; }
    #qr-wrap { margin-top:20px; text-align:center; display:none; }
    #qr-wrap img { width:180px; border-radius:8px; }
    #qr-url { font-size:0.75rem; color:#aaa; word-break:break-all; margin-top:8px; }
  </style>
</head>
<body>
  <div id="video-col">
    <h1>NoCall 🛒</h1>
    <img id="stream" src="/stream" alt="Camera stream">
  </div>
  <div id="cart-col">
    <div id="cart-box">
      <h2 style="margin:0 0 12px;font-size:1rem;">장바구니</h2>
      <table>
        <thead><tr><th>상품</th><th>수량</th><th>금액</th></tr></thead>
        <tbody id="cart-body"></tbody>
        <tfoot><tr id="total-row"><td>합계</td><td></td><td id="total-cell">₩0</td></tr></tfoot>
      </table>
      <button class="btn" id="btn-checkout">결제 QR 생성</button>
      <button class="btn" id="btn-reset">카트 초기화</button>
      <div id="qr-wrap">
        <img id="qr-img" src="" alt="QR">
        <div id="qr-url"></div>
      </div>
    </div>
  </div>
<script>
async function refreshCart() {
  const r = await fetch('/cart');
  const data = await r.json();
  const tbody = document.getElementById('cart-body');
  tbody.innerHTML = '';
  for (const item of data.items) {
    tbody.innerHTML += `<tr><td>${item.name}</td><td>${item.qty}</td><td>₩${item.subtotal.toLocaleString()}</td></tr>`;
  }
  document.getElementById('total-cell').textContent = '₩' + data.total.toLocaleString();
}

document.getElementById('btn-checkout').addEventListener('click', async () => {
  const r = await fetch('/checkout', {method:'POST'});
  if (!r.ok) { alert('카트가 비어있습니다'); return; }
  const url = r.headers.get('X-Receipt-URL');
  const blob = await r.blob();
  const qrWrap = document.getElementById('qr-wrap');
  document.getElementById('qr-img').src = URL.createObjectURL(blob);
  document.getElementById('qr-url').textContent = url;
  qrWrap.style.display = 'block';
});

document.getElementById('btn-reset').addEventListener('click', async () => {
  await fetch('/cart/reset', {method:'POST'});
  document.getElementById('qr-wrap').style.display = 'none';
  refreshCart();
});

setInterval(refreshCart, 1000);
refreshCart();
</script>
</body>
</html>
"""
