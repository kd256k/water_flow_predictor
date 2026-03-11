# water_flow_predictor — 배수지 수요예측 AI 시스템

배수지 유출유량을 LSTM Seq2Seq + Bahdanau Attention으로 예측

## 주요 성과 

- **12개 아키텍처 비교** → Seq2SeqAttn 선정 (MAPE 2.79%)
- **Ablation Study** → Decoder dropout이 v2 성능 2배 악화의 지배적 원인임을 확정
- **최종 모델** → 3-run 평균 MAPE **3.19 ± 0.10%**, seed간 MAPE 표준편차 68% 축소

## 시스템 구성

- **MySQL**: 수도 유량 원본 데이터 및 날씨 데이터를 보관하는 소스 DB
- **flow-api** (port 8000): MySQL에서 데이터를 조회하여 Seq2Seq 모델로 추론 후 결과를 Redis에 저장
- **Redis**: 예측 결과 캐시 (클라이언트 응답용)

데이터 흐름: `MySQL → flow-api → Redis`

## 디렉토리 구조
```
├── src/                    # FastAPI 서빙 코드
│   ├── main.py             # FastAPI 엔트리포인트
│   ├── generator.py        # DB 조회 + 예측 파이프라인
│   ├── inference.py        # 모델 추론 서비스
│   ├── flowpredictor.py    # LSTM 모델 클래스
│   └── seq2seq_predictor.py # Seq2Seq+Attention 모델 클래스
├── notebook/               # 실험 노트북
│   ├── experiment/         # 01~20번: 12개 모델 실험(Phase 1) + v2 개선 시도(Phase 2) + Ablation A/B/C(Phase 3~4)
│   ├── seq2seq_attn_flow.ipynb  # 배포 버전 학습 노트북
│   └── seq2seq_attn_flow.py     # 배포 버전 학습 스크립트
├── models/                 # 학습된 모델 가중치
│   ├── j_resv_seq2seq.pth  # J배수지 모델
│   ├── a_resv_seq2seq.pth  # A배수지 모델
│   └── d_resv_seq2seq.pth  # D배수지 모델
├── data/                   # 원천 데이터 (git 제외)
├── docs/                   # 분석 보고서
│   ├── experiment_report.md
│   ├── 12model_benchmark_report.md
│   ├── segment_id_bug_report.md
│   └── ...                 # 기타 분석 문서
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 실험 요약

| Phase | 내용 | 결과 |
|-------|------|------|
| 1 | 12개 아키텍처 벤치마크 | segment_id 미적용 버그 발견(window 25.56% 오염) → segment-aware 파이프라인 재실험 → Seq2SeqAttn 1위 (MAPE 2.79%) |
| 2 | Best Model 개선 시도 (v2) | 8가지 동시 변경 → 2배 악화 |
| 3 | Ablation Study (A/B) | Decoder dropout = 지배적 원인 |
| 4 | Ablation C | 차분 rainfall 우위 확정 (동일 환경 기준 +0.40%p) |
| 배포 | Scheduler/ES 안정화 | MAPE 3.19±0.10%, seed간 MAPE 표준편차 68% 축소 |

상세 실험 결과는 [`docs/experiment_report.md`](docs/experiment_report.md) 참조.

## 모델 구조
```
Encoder:  LSTM(10→128, 2-layer, dropout=0.2)
Attention: Bahdanau Additive (동적 Query)
Decoder:  LSTMCell(144→128) × 15 steps  # 144 = context(128) + step_embed(16)
Loss:     Step-weighted MSE (1.0 → 2.0)
Params:   377,585개
```

- Input: 72분 (10 features: 유량, 기온, 강수, 습도, 시간/요일/계절 sin/cos)
- Output: 4회 연속 15분 예측 = 총 60분 예측

## 실행 방법

### Docker (권장)
```bash
# .env 파일 생성
cp .env.example .env
# .env 내 DB/Redis 접속 정보 수정

# API 서비스 실행
docker compose up -d flow-api

# 개발 환경 (Jupyter 포함)
docker compose --profile dev up -d
```

### 로컬 실행
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## 기술 스택

- **ML**: PyTorch 2.10, CUDA 12.x
- **API**: FastAPI, Uvicorn
- **Infra**: Docker, Redis, MySQL
- **Data**: pandas, scikit-learn, SciPy (Savitzky-Golay)

## 라이센스

© 2024-2025. All Rights Reserved.

본 프로젝트는 포트폴리오 목적으로 공개되었으며, 무단 복제 및 상업적 사용을 금합니다.
