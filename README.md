# jolup
졸업논문 실증분석 준비 레포

## 구성
- `common/`: 공용 유틸 (데이터수집 헬퍼, 변수변환, 통계검정, 시각화)
- `scripts/collect_binance_data.py`: funding(기본 auto: Binance 실패 시 Bybit fallback) + Binance 가격 데이터 수집
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
python scripts/build_eth_yield_panel.py --cf-srr-csv data/raw/cf_eth_srr.csv
jupyter lab
```

## 워크플로
1. `scripts/collect_binance_data.py` 로 데이터 수집
2. `data/raw/cf_eth_srr.csv` 준비 후 `python scripts/build_eth_yield_panel.py` 실행
   - 선택: staking ratio까지 만들려면 `--with-ratio --staked-csv ... --supply-csv ...`
3. 실험 노트북은 `data/processed/eth_yield_panel.csv`를 우선 사용
   - 없으면 `data/raw/eth_staking_yield_daily.csv`로 fallback
4. 실험별 notebook에서 전처리 + 통계검정 수행
5. 공용 함수는 `common/`에 추가해 재사용

> Binance 선물 API가 국가/환경에 따라 HTTP 451을 반환할 수 있습니다. 이 경우 `--funding-source bybit` 또는 기본 `auto` 모드를 사용하세요.
