from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"

for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)
