# jolup
졸업논문 실증분석 준비 레포

## 구성
- `common/`: 공용 유틸 (데이터수집 헬퍼, 변수변환, 통계검정, 시각화)
- `common/yield_pipeline.py`: ETH staking yield CSV 정규화, Staking Rewards/Lido 수집, staking ratio 패널 생성
- `scripts/collect_binance_data.py`: funding(기본 auto: Binance 실패 시 Bybit fallback) + Binance 가격 데이터 수집
- `scripts/build_eth_yield_panel.py`: ETH staking yield 원천 데이터를 `data/processed/eth_yield_panel.csv`로 변환
- `experiments/01_funding_vs_staking/01_funding_vs_staking.ipynb`: 가설 1/2
- `experiments/02_lag_effect/02_lag_effect.ipynb`: 가설 3
- `experiments/03_carry_gap_mean_reversion/03_carry_gap_mean_reversion.ipynb`: 가설 4
- `notebooks/shared_utils.ipynb`: 공용 시각화/유틸 확인용 노트북
- `data/raw`, `data/processed`: 원천/가공 데이터 저장 폴더

## 빠른 시작
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/collect_binance_data.py --funding-source auto
python scripts/build_eth_yield_panel.py --yield-source assumed --assumed-apr 3.0 --start-date 2023-11-27
jupyter lab
```

## ETH staking yield 데이터 준비

논문 메인 변수는 `StakeYield_t`이며, 파이프라인 표준 컬럼은 `date, stake_yield`입니다.

### 선택지 A: CF ETH_SRR 수동/라이선스 데이터 (논문 메인 권장)
CF Benchmarks 공개 페이지는 지표 설명과 최신 범위를 제공하지만, 역사적 데이터 이용은 별도 라이선스/문의가 필요한 형태입니다. 확보한 CF ETH_SRR 파일을 아래 형식으로 저장하세요.

```csv
date,eth_srr
2023-11-27,0.0342
2023-11-28,0.0339
```

실행:
```bash
python scripts/build_eth_yield_panel.py --yield-source manual --cf-srr-csv data/raw/cf_eth_srr.csv
```

### 선택지 B: 고정 APR 가정 시나리오 (라이선스 없이 바로 실행 가능)
CF ETH_SRR 히스토리를 확보하기 어렵다면, 학사논문 단계에서는 `StakeYield_t = 3%` 같은 상수 가정으로 먼저 파이프라인과 분석을 돌릴 수 있습니다. 이 결과는 “실제 CF ETH_SRR 실증 결과”가 아니라 **가정 기반 민감도 분석**으로 표기해야 합니다.

예: 전체 기간 ETH staking APR을 3.0%로 가정

```bash
python scripts/build_eth_yield_panel.py \
  --yield-source assumed \
  --assumed-apr 3.0 \
  --start-date 2023-11-27 \
  --cf-srr-csv data/raw/cf_eth_srr.csv
```

2.5%, 3.0%, 3.5%처럼 여러 가정으로 반복 실행하면 sensitivity check로 쓸 수 있습니다.

### 선택지 C: Staking Rewards API 히스토리 (자동 수집 proxy)
Staking Rewards API key가 있으면 ETH `reward_rate` 일간 히스토리를 받아 같은 패널로 변환합니다.

```bash
export STAKING_REWARDS_API_KEY="YOUR_API_KEY"
python scripts/build_eth_yield_panel.py \
  --yield-source stakingrewards \
  --start-date 2023-11-27 \
  --cf-srr-csv data/raw/cf_eth_srr.csv
```

### 선택지 D: Lido current APR (파이프라인 smoke test용)
히스토리 회귀분석용이 아니라, 파일 생성/노트북 로딩이 되는지만 확인할 때 사용합니다.

```bash
python scripts/build_eth_yield_panel.py --yield-source lido-current --cf-srr-csv data/raw/cf_eth_srr.csv
```

### Staking ratio 보조 변수
`staking_ratio = staked_eth / total_supply`를 추가하려면 다음 파일을 준비합니다.

- `data/raw/staked_eth_daily.csv`: `date, staked_eth`
- `data/raw/eth_total_supply_daily.csv`: `date, total_supply`

예시 파일은 `data/raw/*.example.csv`에 있습니다.

```bash
python scripts/build_eth_yield_panel.py \
  --yield-source manual \
  --cf-srr-csv data/raw/cf_eth_srr.csv \
  --with-ratio \
  --staked-csv data/raw/staked_eth_daily.csv \
  --supply-csv data/raw/eth_total_supply_daily.csv
```

## 워크플로
1. `scripts/collect_binance_data.py` 로 funding/가격 데이터 수집
2. 위 선택지 중 하나로 `data/processed/eth_yield_panel.csv` 생성
   - 라이선스/API가 없으면 `--yield-source assumed --assumed-apr 3.0`으로 먼저 실행
3. 실험 노트북은 `data/processed/eth_yield_panel.csv`를 우선 사용
   - 없으면 `data/raw/eth_staking_yield_daily.csv`로 fallback
4. 실험별 notebook에서 전처리 + 통계검정 수행
5. 공용 함수는 `common/`에 추가해 재사용

> Binance 선물 API가 국가/환경에 따라 HTTP 451을 반환할 수 있습니다. 이 경우 `--funding-source bybit` 또는 기본 `auto` 모드를 사용하세요.
