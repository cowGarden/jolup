# Experiments/Scripts 진행 현황 요약

이 문서는 현재 레포지토리에서 `experiments/`와 `scripts/`의 동작 상태를 빠르게 파악하고, 다음 단계(예: MA/연도별 분석 추가)를 바로 진행할 수 있도록 정리한 핸드오프 메모다.

## 1) 현재까지 반영된 변경사항

- `experiments` 3개 노트북의 데이터 로딩 경로를 최신 수집 스크립트 산출물과 호환되도록 정리함.
  - 대상:
    - `experiments/01_funding_vs_staking/01_funding_vs_staking.ipynb`
    - `experiments/02_lag_effect/02_lag_effect.ipynb`
    - `experiments/03_carry_gap_mean_reversion/03_carry_gap_mean_reversion.ipynb`
- 노트북에서 funding/price 파일 탐색 우선순위를 다음처럼 통일함.
  - Funding:
    - 신규: `data/raw/binance_funding_ETHUSDT.csv`, `data/raw/binance_funding_BTCUSDT.csv`
    - fallback: 구 포맷(`binance_ethusdt_funding.csv` 등)
  - Price:
    - 신규: `data/raw/binance_klines_ETHUSDT_1d.csv`, `data/raw/binance_klines_BTCUSDT_1d.csv`
    - fallback: 구 포맷(`binance_ethusdt_1d.csv` 등)
  - Staking yield:
    - 우선 `data/processed/eth_yield_panel.csv`
    - fallback `data/processed/lido_wsteth_share_rate.csv`
    - fallback `data/raw/eth_staking_yield_daily.csv`
- `scripts/collect_perp_funding_panel.py`의 Lido 후보 파일 목록에 `data/processed/lido_wsteth_share_rate.csv`를 추가함.
  - 기존에는 `eth_yield_panel.csv`가 없으면 `stake_yield`를 못 붙여서 master 패널에 컬럼 누락이 발생 가능했음.

## 2) 현재 확인된 이슈와 맥락

- 분석 스크립트 실행 시 `stake_yield` 누락 에러가 발생한 이력:
  - `scripts/analyze_eth_btc_funding_staking.py`는 `stake_yield`를 필수 컬럼으로 요구함.
  - 과거 master 생성 시 Lido 후보 경로 미스매치로 `stake_yield` 없이 저장된 파일이 있었음.
  - 현재는 `collect_perp_funding_panel.py`에서 후보 경로 보강하여 재발 가능성은 낮아짐.
- 경고 메시지:
  - `[WARN] Lido staking yield stake_yield max abs=0.117656; rate may be percent-scale.`
  - 의미: 에러가 아니라 스케일 휴리스틱 경고(임계값 0.05 초과 시 경고).
  - 실제 데이터(`lido_wsteth_share_rate.csv`)에 일시적으로 높은 APR 포인트(예: 약 11.76%)가 있어 경고 조건 충족.

## 3) experiments/ 상태

### 공통

- 3개 노트북 모두 `PROJECT_ROOT` 자동 탐색 로직이 있고, 현재 경로 탐색 방식은 일관됨.
- 3개 노트북 모두 최신 수집 결과 파일명과 구 파일명을 함께 지원하도록 수정됨.
- staking 소스는 `eth_yield_panel.csv` 우선 사용으로 통일되어 있음.

### 개별 목적

- `01_funding_vs_staking`: 기본 상관/회귀 검정 (가설1/2).
- `02_lag_effect`: lag effect 중심 검정 (가설3).
- `03_carry_gap_mean_reversion`: carry-gap 평균회귀 검정 (가설4).

## 4) scripts/ 상태

- `scripts/collect_perp_funding_panel.py`
  - 역할: funding/OI/price/optional controls + staking을 결합해 master 패널 생성.
  - 핵심 출력: `data/processed/master_{exchange}_eth_btc_funding_staking_daily.csv`
  - 현재 상태: `lido_wsteth_share_rate.csv`도 자동 후보로 인식하도록 보완됨.
- `scripts/build_eth_yield_panel.py`
  - 역할: 다양한 소스(`manual`, `assumed`, `lido-rpc`, `stakingrewards`, `lido-current`)로 yield 패널 생성.
  - 권장 출력: `data/processed/eth_yield_panel.csv` (파이프라인 표준 입력으로 사용 권장).
- `scripts/analyze_eth_btc_funding_staking.py`
  - 역할: master 패널 검증 + OLS/HAC + 강건성/그룹/carry-gap/pooled 분석.
  - 제약: `stake_yield` 필수.
- `scripts/collect_binance_data.py`
  - 구 수집 파이프라인 성격(신규 파일명 체계와 일부 차이).
- `scripts/plot_lido_yield_panel.py`
  - yield 시계열 시각화 용도.

## 5) 재현 가능한 실행 순서 (권장)

1. Yield 패널 생성(표준 파일명):
   - `python scripts/build_eth_yield_panel.py --yield-source lido-rpc --ethereum-rpc-url <RPC_URL> --out-csv data/processed/eth_yield_panel.csv`
2. Perp/funding master 패널 생성:
   - `python scripts/collect_perp_funding_panel.py --exchange binance`
   - 필요 시 명시: `--lido-yield-csv data/processed/lido_wsteth_share_rate.csv`
3. 분석 실행:
   - `python scripts/analyze_eth_btc_funding_staking.py`
4. 노트북(`experiments`) 재실행:
   - 상단 데이터 로딩 셀부터 순서대로 재실행.

## 6) Codex에 바로 넘길 다음 작업(요청사항 반영)

### A. MA(이동평균) 확장

- 대상:
  - `scripts/analyze_eth_btc_funding_staking.py`의 rolling/lag 관련 섹션
  - 필요 시 `experiments/02_lag_effect`, `experiments/03_carry_gap_mean_reversion`에도 동일 로직 반영
- 제안:
  - `stake_yield_ma{7,14,30}` 외 `spread_ma{7,14,30}` 추가
  - MA 조합 회귀 스펙(`yield_ma vs spread`, `yield_ma + controls`) 템플릿화
  - 결과표 파일명 규칙 통일 (`*_ma_specs.csv`)

### B. 연도별 동작(Yearly regime) 추가

- 목적:
  - 연도별 계수 안정성/유의성 비교
  - 이벤트(ETF 전후)와 연도 교차효과 확인
- 제안:
  - `year` 컬럼 파생 후 연도별 분할 회귀 루프
  - 최소 관측치 미달 연도 skip + 경고
  - 출력:
    - `*_yearly_regression_summary.csv`
    - `*_yearly_quintile_summary.csv`
  - 그래프:
    - 연도별 `stake_yield` 계수/CI 바차트

### C. 방어적 개선(선택)

- `analyze_eth_btc_funding_staking.py`에서 `stake_yield`가 없을 때:
  - `annualized_apr_decimal` -> `stake_yield` 자동 매핑(있다면)
  - 둘 다 없으면 명확한 에러와 해결 가이드 출력

## 7) 현재 리스크/체크포인트

- `lido-rpc` 기반 산출치에 단일 고APR 점(이상치 가능)이 존재할 수 있음.
  - winsorize/clip/robust regression 민감도 점검 권장.
- 구 수집 스크립트(`collect_binance_data.py`)와 신규 스크립트(`collect_perp_funding_panel.py`)의 파일명 체계가 혼재되어 있음.
  - 신규 체계 기준으로 점진적 정리 필요.

---

최소 결론:
- 현재 `experiments`는 최신/구 파일명을 모두 수용하도록 맞춰졌고,
- `collect_perp_funding_panel.py`도 `lido_wsteth_share_rate.csv`를 인식하도록 보강되어,
- 다음 단계인 **MA/연도별 분석 확장 작업을 바로 시작할 수 있는 상태**다.
