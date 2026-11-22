from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PICTURE_DIR = BASE_DIR / "picture_data"
METADATA_DIR = PICTURE_DIR / "metadata"

META_CSV_PATH = METADATA_DIR / "search_results.csv"

