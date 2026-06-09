# HANDOFF — for Codex (AI coding agent)

> 이 문서는 **AI 코딩 에이전트(Codex)** 가 직접 읽고 실행하도록 작성됨.
> 사람용 안내는 `검증_사용설명서_비개발자용.md`, 개발자용은 `train/VERIFY.md`.
> Codex 사용 시 이 파일을 `AGENTS.md` 로 복사해 두면 자동 컨텍스트로 잡힌다.

## 0. TL;DR (에이전트 작업 지시)
기본 임무는 **재학습이 아니라 검증(verify)** 이다. 다음을 수행하라:
1. 환경 셋업 (`git lfs pull` → `uv sync`)
2. 수치 검증 `uv run python train/verify.py` → 5클래스 PASS 확인
3. 실동작 검증 `uv run python run.py` → `http://localhost:8000` (웹캠)
4. 결과를 아래 "보고 형식"으로 요약
사용자가 명시적으로 요청하기 전에는 **재학습/커밋/푸시 금지**.

---

## 1. 프로젝트 개요
- **무엇**: YOLOv8 기반 5-class 객체탐지로 "스마트 계산대"(계산대 위 한 겹으로 펼친 상품을 클래스별 개수로 집계 → 영수증) 데모. FastAPI 웹앱.
- **클래스 (ID 고정)**: `0 cola · 1 water · 2 cereal-box · 3 paper-cup · 4 cup-noodle`
  - 이 매핑은 `train/nocall.yaml`, `train/merge.py`, `app/cart.py` 에서 **반드시 일치**해야 함. 변경 시 세 곳 동시 수정.
- **모델 파일**: `models/nocall.pt` (약 21MB, **Git LFS** 로 추적). 2단계 학습 결과.
- **현재 성능 기준선**: 1단계 pretrain mAP50 0.981 → 2단계 finetune mAP50 0.993. `verify.py` 기준 5클래스 모두 PASS(평균 conf 0.82~0.90).

## 2. 리포 맵
```
run.py                      # 앱 진입점 (uvicorn :8000)
app/
  main.py                   # FastAPI: / , /stream(MJPEG), /cart, /checkout, /receipt
  camera.py                 # 웹캠 + YOLO detect + 박스 오버레이, 프레임별 개수 집계 (conf=0.30)
  cart.py                   # PRODUCTS(클래스→이름·가격), 개수 집계 + 최근 N프레임 중앙값 평활화
models/nocall.pt            # 학습된 모델 (LFS)
train/
  nocall.yaml               # 데이터셋 정의. path: PLACEHOLDER (런타임 치환)
  download.py               # Roboflow 5개 데이터셋 다운로드
  auto_label.py             # Grounding DINO 자동 라벨 (optional deps)
  dedup_labels.py           # NMS 중복 bbox 제거
  augment.py                # albumentations ×8 오프라인 증강 (custom)
  merge.py                  # 통합. 모드: pretrain | finetune
  train.py                  # 1단계 사전학습 (yolov8s, scratch)
  finetune.py               # 2단계 미세조정 (freeze backbone, 낮은 lr)
  verify.py                 # 모델 검증 (웹캠 불필요)
  VERIFY.md / HANDOFF_*.md  # 문서
1_모델검증.bat / 2_데모실행.bat   # 비개발자용 더블클릭 런처
```

## 3. 환경 셋업 (정확한 명령)
전제: `git`, `git-lfs`, `uv` 설치됨. Python은 uv가 3.12 자동 관리.
```bash
git lfs install
git lfs pull                 # ★ 모델 실파일 수신 (안 하면 포인터 텍스트만 옴)
uv sync                      # 기본 의존성 (라벨링 deps 제외)
```
검증: `models/nocall.pt` 크기가 **~21MB** 여야 정상. 수백 바이트면 LFS 미수신 → `git lfs pull` 재실행.

플랫폼 주의 (pyproject `tool.uv.sources`):
- **Windows**: torch가 CUDA cu128 휠로 설치됨 (NVIDIA GPU 사용).
- **macOS**: PyPI 기본 torch (MPS/CPU). 코드가 device 자동 선택(cuda→mps→cpu).

## 4. 검증 절차
### 4-1. 수치 검증 (웹캠 불필요, CI 친화적)
```bash
uv run python train/verify.py            # 기본: 저장된 custom 웹캠 이미지로 추론
uv run python train/verify.py <폴더>      # 임의 테스트 이미지 폴더 지정
```
- 출력: 클래스별 검출수·평균 신뢰도·PASS/CHECK, "검출 0개" 수.
- **합격**: 5클래스 모두 PASS(평균 conf ≥ 0.50). 결과 이미지 `runs/verify/`.
- 주의: 기본 이미지는 학습 도메인이라 점수가 높음. **일반화 확인은 새 이미지 폴더**로.

### 4-2. 실동작 검증 (웹캠)
```bash
uv run python run.py
# 브라우저: http://localhost:8000
```
- 5개 제품을 비춰 박스에 `클래스명 + %` 가 정확히(70%+) 뜨는지.
- 계산대 위에 **서로 안 겹치게 한 겹으로** 펼친 상품이 클래스별 개수로 장바구니에 집계되는지. (잠깐 깜빡여도 약 1초 평활화로 유지)
- 헤드리스/원격 환경이면 웹캠 불가 → 4-1만 수행하고 그 사실을 보고.

## 5. 알려진 함정 (이번 세션에서 확인됨 — 재발견 불필요)
1. **Git LFS 모델**: clone/pull 후 `git lfs pull` 안 하면 `models/nocall.pt` 가 포인터 텍스트. `verify.py`가 "모델 없음/손상" 안내.
2. **storage.googleapis.com TLS 차단**: 일부 네트워크(한국 SNI 필터/백신 HTTPS 검사)에서 `download.py`(Roboflow)가 `SSL record layer failure`로 실패. → **모바일 핫스팟/VPN** 으로 우회. (재다운로드할 때만 해당. 검증엔 무관.)
3. **Roboflow 폴더 skip**: 대상 폴더가 (빈 채로라도) 존재하면 라이브러리가 다운로드를 건너뜀. 빈 폴더 삭제 후 재시도.
4. **GPG 커밋 서명 타임아웃**: 이 환경은 `commit.gpgsign=true`. pinentry 미동작 시 커밋 실패. 커밋이 꼭 필요하면 에이전트가 임의로 `--no-gpg-sign` 하지 말고 **사용자에게 확인**.
5. **라벨링 의존성은 optional**: autodistill/grounding-dino/transformers/scikit-learn 은 메인 deps 아님 → `uv sync --extra label` 로만 설치. 핀 고정 이유: `transformers==4.44.2`(GroundingDINO 호환, 5.x는 `get_head_mask` 없음), `scikit-learn==1.5.2`(Windows 앱제어가 최신 DLL 차단).
6. **albumentations 2.x**: `GaussNoise(std_range=...)`, `Cutout→CoarseDropout` (augment.py 이미 반영).
7. **autodistill 출력 구조**: 중첩 데이터셋으로 떨어짐 → auto_label.py가 평면화함. 라벨 class id는 0이지만 merge.py가 remap하므로 무관.
8. **증강물 비커밋**: `aug_*` 는 .gitignore 처리(augment.py로 재생성). `train/data/`도 비커밋(단 `*-custom` 원본+라벨은 추적).
9. **nocall.yaml `path: PLACEHOLDER`**: train.py/finetune.py가 런타임에 절대경로로 치환(임시 yaml). **직접 하드코딩 금지.** verify.py/run.py는 path 무관.

## 6. (요청 시에만) 재학습 — 2단계 파이프라인
사용자가 재학습을 명시하면 순서대로:
```bash
# 데이터
uv run python train/download.py            # Roboflow (TLS 차단 시 핫스팟)
uv sync --extra label                        # 라벨링 deps
uv run python train/auto_label.py            # Grounding DINO 자동 라벨(+LabelImg 수동검수: 대화형)
uv run python train/dedup_labels.py          # NMS 중복 제거
uv run python train/augment.py               # custom ×8 증강
# 1단계: 사전학습 (Roboflow only, paper-cup cap 800)
uv run python train/merge.py pretrain
uv run python train/train.py                 # → models/nocall.pt (pretrained)
# 2단계: 미세조정 (custom 증강 + Roboflow 닻, backbone freeze)
uv run python train/merge.py finetune
uv run python train/finetune.py              # → models/nocall.pt (최종)
```
- GPU 권장(RTX). batch 16이 6GB에서 ~3.6G로 들어감(OOM시 train.py/finetune.py BATCH=8).
- 1단계 ~1시간, 2단계 ~15분.
- 클래스 불균형: paper-cup(3,181)을 pretrain에서 800으로 다운샘플(`merge.py PRETRAIN_CAP`).

## 7. 보고 형식 (에이전트 출력 권장 템플릿)
```
[환경] OS / GPU(cuda·mps·cpu) / 모델 크기(MB, LFS 정상여부)
[수치검증] 클래스별 PASS/CHECK 표 + 검출0 수 + 종합 PASS/CHECK
[실동작] (가능 시) 5클래스 인식·개수 집계 정확 여부 / (웹캠 불가 시) 사유
[이슈] 발생한 함정과 처리
[결론] 검증 통과 여부 + 권장 후속조치
```

## 8. 하지 말 것
- 사용자 승인 없이 재학습·커밋·푸시·force push.
- `--no-gpg-sign` 임의 사용(§5-4).
- `nocall.yaml` 경로 하드코딩, 클래스 매핑 단독 변경(§1).
- `.env`/`ROBOFLOW_API_KEY` 등 비밀값 노출·커밋.
- `train/data/`·`aug_*`·대용량 파일 커밋.
