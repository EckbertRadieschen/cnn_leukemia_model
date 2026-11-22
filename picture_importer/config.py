from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PICTURE_DATA_DIR = BASE_DIR / "picture_data"
DATASET_DIR = PICTURE_DATA_DIR / "dataset"

REVIEW_DIR = DATASET_DIR.parent / "review"

METADATA_DIR = PICTURE_DATA_DIR / "metadata"
METADATA_FILE = METADATA_DIR / "image_metadata.csv"

CLASSES = [
    "tasse",
    "geldboerse",
    "flasche",
    "brille",
    "schuh",
    "handy",
    "hund",
    "kirche"
]

SEARCH_TERMS = {
    "brille": [
        "eyeglasses",
        "spectacles",
        "glasses worn",
        "glasses isolated"
    ],
    "schuh": [
        "shoes",
        "sneakers",
        "footwear",
        "shoes worn"
    ],
    "hund": [
        "dog portrait",
        "dog standing",
        "dog running",
        "dog playing"
    ],
    "kirche": [
        "church exterior",
        "church interior",
        "church building",
        "cathedral"
    ],
    "flasche": [
        "water bottle",
        "plastic bottle",
        "bottle on table",
        "person holding bottle"
    ],
    "geldboerse": [
        "leather wallet",
        "purse wallet",
        "wallet with money",
        "person holding wallet"
    ],
    "handy": [
        "smartphone",
        "smartphone in hand",
        "person using smartphone",
        "smartphone on table"
    ],
    "tasse": [
        "coffee cup",
        "tea cup",
        "ceramic mug",
        "person holding cup"
    ]
}

SEARCH_LIMIT_PER_TERM = 150

SPLITS = [
    "train",
    "validation",
    "test"
]

IMAGE_SIZE = (224, 224)
MIN_IMAGE_WIDTH = 224
MIN_IMAGE_HEIGHT = 224

REQUEST_DELAY_SECONDS = 2
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 5