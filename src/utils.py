from pathlib import Path

RANDOM_SEED = 42
MARKET_OPEN_H, MARKET_OPEN_M = 9, 30
MARKET_CLOSE_H, MARKET_CLOSE_M = 16, 0

TRAIN_CUTOFF = "2022-12-31"
VAL_CUTOFF = "2023-12-31"
TEST_START = "2024-01-01"

# directory structure
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw_data"
CLEAN_DIR = ROOT_DIR / "data" / "clean_data"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
NOTE_DIR = ROOT_DIR / "notebook"
NOTE_DIR.mkdir(parents=True, exist_ok=True)

# input files
OFFERINGS_FILES = [
    RAW_DIR / "temp_offerings_2021_anon.tsv",
    RAW_DIR / "temp_offerings_2022_anon.tsv",
    RAW_DIR / "temp_offerings_2023_anon.tsv",
    RAW_DIR / "temp_offerings_2024_anon.tsv",
]
PRICES_FILE = RAW_DIR / "temp_prices_2021_2024_anon.tsv"

MASTER_INDEX_FILE = CLEAN_DIR / "master_index.csv"
FEATURES_FILE = CLEAN_DIR / "features.npz"
META_FILE = CLEAN_DIR / "feature_meta.csv"