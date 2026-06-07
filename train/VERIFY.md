# 모델 검증 파이프라인 (팀원 전송용)

학습된 5-class 모델(`models/nocall.pt`)이 제대로 동작하는지 **받는 쪽에서 검증**하는 절차.
재학습이 아니라 **검증**이 목적. 약 10분 소요.

대상 클래스: `cola` / `water` / `cereal-box` / `paper-cup` / `cup-noodle`

---

## 0. 사전 준비 (기기당 1회)

```bash
# git-lfs 필수 — 모델이 LFS로 저장돼 있음
#   Windows: winget install GitHub.GitLFS   |   Mac: brew install git-lfs
git lfs install

git clone git@github.com:JunyoungCho07/nocall-prototype.git   # 또는 git pull && git lfs pull
cd nocall-prototype
uv sync                       # 의존성 설치 (CUDA torch 포함, Windows)
```

- 모델 파일 확인: `models/nocall.pt`가 **약 21MB**면 정상. 몇백 바이트면 LFS 미설치 → `git lfs pull`.

---

## 1. 모델 자체 검증 (웹캠 불필요)

```bash
uv run python train/verify.py
```

- 저장소에 포함된 웹캠 커스텀 이미지로 추론 → **클래스별 검출 수·평균 신뢰도**를 출력
- 박스 그린 결과가 `runs/verify/`에 저장됨 → 눈으로 확인
- **합격 기준**: 5개 클래스 모두 `PASS` (평균 신뢰도 ≥ 0.5), "검출 0개 이미지" 적을수록 좋음

> 참고: 기본 검증 이미지는 학습에 쓰인 도메인 이미지라 점수가 높게 나옴.
> **진짜 일반화 확인**은 새로 찍은 사진 폴더로:  `uv run python train/verify.py <새사진_폴더>`

기준 결과(작성 시점, 학습 도메인 기준):
| 클래스 | 평균 신뢰도 |
|---|---|
| cola | 0.89 |
| water | 0.85 |
| cereal-box | 0.82 |
| paper-cup | 0.90 |
| cup-noodle | 0.89 |

---

## 2. 실제 동작 검증 (웹캠) — 가장 중요

```bash
uv run python run.py
```
→ 브라우저 `http://localhost:8000`

확인 항목:
1. 영상에 제품을 비추면 박스에 **올바른 클래스명 + 신뢰도(%)** 표시
2. 5개 제품 각각을 다양한 각도/거리로 → **이름 정확 & 70%+** 나오는지
3. 화면 중앙 **빨간 스캔라인**을 제품으로 **위→아래** 통과 → 오른쪽 장바구니에 **추가**
4. **아래→위** 통과 → 장바구니에서 **제거**
5. 가격 매핑 정상: 콜라 1,800 / 물 1,000 / 시리얼 6,000 / 종이컵 500 / 컵라면 1,500

---

## 3. 합격 기준 (체크리스트)

- [ ] `verify.py` — 5개 클래스 모두 PASS
- [ ] 앱에서 5개 제품 모두 이름 정확 & 신뢰도 70%+
- [ ] 스캔라인 통과 시 장바구니 추가/제거 정상
- [ ] 오탐(엉뚱한 물체를 제품으로 검출)이 과하지 않음

---

## 4. 문제 해결

| 증상 | 원인 | 조치 |
|---|---|---|
| `모델 없음` / 파일이 수백 바이트 | git-lfs 미설치로 포인터만 받음 | `git lfs install && git lfs pull` |
| 콜라/물 등 인식 안 됨 | 폴백 모델 사용(파일 손상) | `models/nocall.pt` 21MB 확인, 재-pull |
| 특정 클래스만 약함 | 도메인 차이 (조명/제품 다름) | 그 제품 추가 촬영 후 재학습 (HANDOFF_정확도개선.md) |
| 영상 검은 화면 | 웹캠 점유(줌 등) / 인덱스 | 다른 앱 종료, `app/camera.py`의 `VideoCapture(0)` 변경 |
| 신뢰도 전반 낮음 | 임계값 | `app/camera.py`의 `conf=0.30` 조정 |
| `train/nocall.yaml` 경로 오류(학습 시) | PC별 절대경로 | 검증만 할 거면 무관 (train/finetune 시에만 필요) |

---

## 요약 (치트시트)

```bash
git lfs install && git lfs pull        # 모델 받기
uv sync                                 # 환경
uv run python train/verify.py           # ① 모델 검증 (수치)
uv run python run.py                     # ② 실제 동작 (웹캠) → localhost:8000
```
