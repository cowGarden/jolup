# jolup
졸업논문 실증분석 준비 레포

## 구성
- `common/`: 공용 유틸 (데이터수집 헬퍼, 변수변환, 통계검정, 시각화)
- `scripts/collect_binance_data.py`: Binance funding/가격 데이터 수집 스크립트
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
python scripts/collect_binance_data.py
jupyter lab
```

## 워크플로
1. `scripts/collect_binance_data.py` 로 데이터 수집
2. 실험별 notebook에서 전처리 + 통계검정 수행
3. 공용 함수는 `common/`에 추가해 재사용
