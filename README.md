# NoCall — Smart Cart Prototype

> "셀프 계산대인데, 진짜로 셀프로 끝낸다"

계산대 위에 상품을 한 겹으로 펼쳐 놓으면 카메라가 전체를 비춰 자동으로 인식·집계하는 컴퓨터 비전 프로토타입입니다.

## 동작 원리

```
웹캠 (top-down) → YOLOv8 detect → 클래스별 개수 집계 → 프레임 평활화 → 카트 확정 → QR 영수증
```

- 계산대 위에 상품을 **서로 겹치지 않게 한 겹으로** 펼쳐 놓는다
- 매 프레임 **보이는 클래스별 개수**를 그대로 장바구니로 집계
- 최근 N프레임(기본 15, 약 1초)의 **중앙값으로 평활화** → 감지 떨림 방지

## 인식 제품 (5종)

| 클래스 | 제품 | 가격 |
|---|---|---|
| 0 | 콜라 | ₩1,800 |
| 1 | 물 | ₩1,000 |
| 2 | 시리얼(박스) | ₩6,000 |
| 3 | 종이컵 | ₩500 |
| 4 | 컵라면 | ₩1,500 |

## 설치 및 실행

### 요구사항
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- 웹캠

### 환경 설정

```bash
git clone https://github.com/JunyoungCho07/nocall-prototype.git
cd nocall-prototype
uv sync
cp .env.example .env
# .env에 ROBOFLOW_API_KEY 입력
```

### 학습 데이터 준비

```bash
uv run python train/download.py   # Roboflow에서 데이터셋 다운로드
uv run python train/merge.py      # 데이터 통합 및 class ID 재매핑
```

### 모델 학습

```bash
# CUDA (권장 — NVIDIA GPU)
uv run python train/train.py

# Windows (CUDA 자동 감지)
uv run python train/train.py
```

### 서버 실행

```bash
uv run python run.py
# http://localhost:8000 접속
```

## 프로젝트 구조

```
nocall-prototype/
├── app/
│   ├── main.py          # FastAPI 서버 (stream / cart / checkout / receipt)
│   ├── cart.py          # CartManager + 클래스별 개수 집계·평활화
│   ├── camera.py        # OpenCV + YOLOv8 detect
│   └── templates/
│       └── receipt.html # 영수증 페이지
├── train/
│   ├── download.py      # Roboflow 데이터셋 다운로드
│   ├── merge.py         # 멀티 데이터셋 통합
│   ├── train.py         # YOLOv8 fine-tune
│   └── nocall.yaml      # YOLO 데이터셋 설정
├── models/              # 학습된 모델 (nocall.pt) — gitignore
└── run.py               # 서버 실행 진입점
```

## API 엔드포인트

| Method | Path | 설명 |
|---|---|---|
| GET | `/` | 데모 메인 화면 |
| GET | `/stream` | MJPEG 실시간 카메라 스트림 |
| GET | `/cart` | 현재 카트 상태 JSON |
| POST | `/cart/reset` | 카트 초기화 |
| POST | `/checkout` | QR 코드 생성 |
| GET | `/receipt/{id}` | 영수증 페이지 |

## 하드웨어 권장 사양

- **학습**: NVIDIA GPU (RTX 4060 이상) + CUDA
- **추론**: 웹캠 + CPU/GPU (실시간 30fps 가능)
- **카메라**: 외부 웹캠 (Logitech C920 등) + 플렉시블 암 거치대

---

KAIST KEI504 스타트업 마케팅 7조 — NoCall
