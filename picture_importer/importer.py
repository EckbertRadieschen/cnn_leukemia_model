from config import DATASET_DIR, METADATA_DIR, CLASSES, SPLITS

def create_directory_structure():

    for split in SPLITS:
        for class_name in CLASSES:

            directory = DATASET_DIR / split / class_name
            directory.mkdir(parents=True, exist_ok=True)

            print(f"Verzeichnis erstellt: {directory}")

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Metadaten-Verzeichnis erstellt: {METADATA_DIR}")


if __name__ == "__main__":
    create_directory_structure()