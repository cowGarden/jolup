# jolup
졸업논문 실증분석 준비 레포

## 구성
- `common/`: 공용 유틸 (데이터수집 헬퍼, 변수변환, 통계검정, 시각화)
- `common/yield_pipeline.py`: ETH staking yield CSV 정규화, Lido wstETH share-rate 수집, Staking Rewards/Lido 보조 수집, staking ratio 패널 생성
- `scripts/collect_binance_data.py`: funding(기본 auto: Binance 실패 시 Bybit fallback) + Binance 가격 데이터 수집
- `scripts/build_eth_yield_panel.py`: ETH staking yield 원천 데이터를 `data/processed/eth_yield_panel.csv`로 변환
- `scripts/plot_lido_yield_panel.py`: Lido share rate 및 annualized APR 시계열 플롯 저장
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
export ETHEREUM_RPC_URL="https://ethereum.publicnode.com"
python scripts/build_eth_yield_panel.py --yield-source lido-rpc --start-date 2023-11-27
python scripts/plot_lido_yield_panel.py
jupyter lab
```

## ETH staking yield 데이터 준비

논문 메인 변수는 `StakeYield_t`이며, 파이프라인 표준 컬럼은 `date, stake_yield`입니다. 본 프로젝트의 기본 권장값은 **validator gross yield**가 아니라 **Lido wstETH/stETH protocol exchange rate 기반 retail-accessible staking yield proxy**입니다.

### 선택지 A: Lido wstETH share-rate via Ethereum RPC (무료·재현 가능, 기본 권장)

wstETH는 non-rebasing wrapper이므로 1 wstETH가 나타내는 stETH 수량(`stEthPerToken`)이 시간이 지날수록 증가합니다. 이 protocol exchange rate는 2차 시장가격이 아니므로 stETH/ETH depeg, wstETH/ETH premium/discount를 yield와 섞지 않습니다.

정의:

```text
ExchangeRate_t = stETH_per_wstETH_t
DailyYield_t = ExchangeRate_t / ExchangeRate_{t-1} - 1
AnnualizedAPR_t = DailyYield_t * 365
```

실행:

```bash
export ETHEREUM_RPC_URL="https://ethereum.publicnode.com"
python scripts/build_eth_yield_panel.py \
  --yield-source lido-rpc \
  --start-date 2023-11-27 \
  --cf-srr-csv data/raw/lido_wsteth_share_rate.csv \
  --out-csv data/processed/eth_yield_panel.csv
```

출력 컬럼:
- `date`: UTC daily date
- `block_number`, `block_timestamp_utc`: 해당 날짜 샘플링에 사용된 Ethereum block
- `share_rate`, `exchange_rate`: stETH per 1 wstETH
- `daily_yield_decimal`: 일간 share-rate 증가율
- `annualized_apr_decimal`: `daily_yield_decimal * 365`
- `annualized_apr_pct`: percent 단위 APR
- `stake_yield`: 회귀에서 사용하는 연율 decimal 값 (`annualized_apr_decimal`)

주의:
- 이 값은 restaking yield를 포함하지 않습니다.
- 이 값은 validator gross yield가 아니라 Lido LST holder가 접근 가능한 net-ish proxy입니다.
- stETH/ETH market price ratio는 depeg/basis risk 변수이지 staking yield 자체가 아닙니다.

### 선택지 B: 고정 APR 가정 시나리오 (라이선스/API 없이 바로 실행 가능)
CF ETH_SRR 히스토리나 RPC 접근이 어렵다면 `StakeYield_t = 3%` 같은 상수 가정으로 먼저 파이프라인과 분석을 돌릴 수 있습니다. 이 결과는 “실제 Lido/CF 실증 결과”가 아니라 **가정 기반 민감도 분석**으로 표기해야 합니다.

```bash
python scripts/build_eth_yield_panel.py \
  --yield-source assumed \
  --assumed-apr 3.0 \
  --start-date 2023-11-27 \
  --cf-srr-csv data/raw/cf_eth_srr.csv
```

2.5%, 3.0%, 3.5%처럼 여러 가정으로 반복 실행하면 sensitivity check로 쓸 수 있습니다.

### 선택지 C: CF ETH_SRR 수동/라이선스 데이터
CF Benchmarks 공개 페이지는 지표 설명과 최신 범위를 제공하지만, 역사적 데이터 이용은 별도 라이선스/문의가 필요한 형태입니다. 확보한 CF ETH_SRR 파일을 아래 형식으로 저장하세요.

```csv
date,eth_srr
2023-11-27,0.0342
2023-11-28,0.0339
```

```bash
python scripts/build_eth_yield_panel.py --yield-source manual --cf-srr-csv data/raw/cf_eth_srr.csv
```

### 선택지 D: Staking Rewards API 히스토리 (자동 수집 proxy)
Staking Rewards API key가 있으면 ETH `reward_rate` 일간 히스토리를 받아 같은 패널로 변환합니다.

```bash
export STAKING_REWARDS_API_KEY="YOUR_API_KEY"
python scripts/build_eth_yield_panel.py \
  --yield-source stakingrewards \
  --start-date 2023-11-27 \
  --cf-srr-csv data/raw/cf_eth_srr.csv
```

### 선택지 E: Lido current APR (파이프라인 smoke test용)
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
  --yield-source lido-rpc \
  --start-date 2023-11-27 \
  --with-ratio \
  --staked-csv data/raw/staked_eth_daily.csv \
  --supply-csv data/raw/eth_total_supply_daily.csv
```

## 워크플로
1. `scripts/collect_binance_data.py` 로 funding/가격 데이터 수집
2. `--yield-source lido-rpc`로 Lido wstETH share-rate 기반 `data/processed/eth_yield_panel.csv` 생성
   - RPC가 없으면 `--yield-source assumed --assumed-apr 3.0`으로 먼저 실행
3. `scripts/plot_lido_yield_panel.py`로 share_rate/APR 시계열 점검
4. 실험 노트북은 `data/processed/eth_yield_panel.csv`를 우선 사용
   - 없으면 `data/raw/eth_staking_yield_daily.csv`로 fallback
5. 실험별 notebook에서 전처리 + 통계검정 수행
6. 공용 함수는 `common/`에 추가해 재사용

> Binance 선물 API가 국가/환경에 따라 HTTP 451을 반환할 수 있습니다. 이 경우 `--funding-source bybit` 또는 기본 `auto` 모드를 사용하세요.

## ETH-BTC perpetual funding spread master panel

`scripts/collect_perp_funding_panel.py` is the full data-collection and model-estimation entry point for the thesis question: whether ETH staking yield is reflected in the ETH-BTC perpetual futures funding spread.

Example smoke-friendly command that avoids the largest optional downloads:

```bash
python scripts/collect_perp_funding_panel.py \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --exchange both \
  --skip-hourly \
  --skip-optional
```

Full command, including hourly realized volatility, optional Binance basis/taker/long-short controls, stETH discount, and OLS-HAC model summaries:

```bash
python scripts/collect_perp_funding_panel.py \
  --start-date 2023-01-01 \
  --exchange both \
  --lido-yield-csv data/processed/lido_staking_yield_daily.csv
```

Main outputs:

- Raw funding: `data/raw/binance_funding_BTCUSDT.csv`, `data/raw/binance_funding_ETHUSDT.csv`, `data/raw/bybit_funding_BTCUSDT.csv`, `data/raw/bybit_funding_ETHUSDT.csv`
- Raw OI: `data/raw/bybit_oi_BTCUSDT.csv`, `data/raw/bybit_oi_ETHUSDT.csv`, plus Binance OI if the API returns enough recent data
- Raw klines: `data/raw/{exchange}_klines_{symbol}_{interval}.csv`
- Daily funding spreads: `data/processed/{exchange}_eth_btc_funding_spread_daily.csv`
- Price/RV/volume controls: `data/processed/{exchange}_price_features_daily.csv`
- OI controls: `data/processed/{exchange}_oi_features_daily.csv`
- Master panels: `data/processed/master_{exchange}_eth_btc_funding_staking_daily.csv`
- OLS-HAC summaries: `data/processed/ols_hac_model_summary_{exchange}.csv` and `data/processed/ols_hac_model_summary_all_exchanges.csv`
- Coverage diagnostics: `data/processed/data_coverage_report.csv`

Implementation notes:

1. Binance funding uses forward pagination with `startTime`, `endTime`, and `next_start = last_fundingTime + 1` so it does not stop after a single 1000-row page.
2. Bybit funding uses backward pagination from `endTime`; each page moves to `oldest_fundingRateTimestamp - 1`, then trims and sorts the final sample in ascending UTC date order.
3. Daily funding is annualized as `sum(intraday fundingRate) * 365`, with rates stored in decimal units.
4. OI controls include `oi_eth_btc = dlog_oi_eth - dlog_oi_btc` and `oi_ratio = log(oi_eth_usd / oi_btc_usd)`. If OI is raw coin/contract quantity, the script multiplies by daily close to construct USD notional.
5. The master panel merges spread, staking yield, price/risk, leverage-demand, liquidity/activity, basis, LST-risk, persistence, and ETF event controls by UTC `date`.
6. Model summaries estimate M1-M8 with OLS-HAC/Newey-West errors and report the sign and significance of the `stake_yield` coefficient.
