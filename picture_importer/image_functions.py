from pathlib import Path
from PIL import Image, ImageOps
import shutil
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

PICTURE_DATA_DIR = BASE_DIR / "picture_data"
REVIEW_DIR = PICTURE_DATA_DIR / "review"
START_BILDER_DIR = PICTURE_DATA_DIR / "start_bilder"
# START_BILDER_DATASET_DIR = PICTURE_DATA_DIR / "start_bilder_dataset"

DATASET_DIR = PICTURE_DATA_DIR / "dataset"
START_BILDER_DATASET_DIR = PICTURE_DATA_DIR / "cancer_images" / "cancer_analysis"
CANCER_SOURCE_DIR = PICTURE_DATA_DIR / "cancer_images" / "Original"

TEST_DIR = DATASET_DIR / "test"
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"


def resize_image(image_path, max_size=(224, 224), quality=90):

    image_path = Path(image_path)

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)

        if image.width <= 150 and image.height <= 150:
            image_path.unlink()
            return

        if image.width <= max_size[0] and image.height <= max_size[1]:
            return

        image.thumbnail(max_size, Image.Resampling.LANCZOS)

        new_image_path = Path(str(image_path).split(".")[0] + ".jpg")

        new_image_path = str(image_path.with_suffix("")) + ".jpg"

        image_path.unlink()

        image.save(
            new_image_path,
            format="JPEG",
            quality=quality,
            optimize=True
        )

def resize_directory(directory: Path):
    for image_path in directory.rglob("*"):
        if str(image_path).endswith(".jpg") or str(image_path).endswith(".jfif"):
            try:
                resize_image(image_path)
                print(f"Verarbeitet: {image_path.name}")

            except Exception as e:
                print(f"Fehler bei {image_path}: {e}")


def count_missed_images (directory: Path, max_size=(224, 224)):
    counter = 0
    
    for image_path in directory.rglob("*"):
        if str(image_path).endswith(".jpg"):

            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image)

                if image.width < max_size[0] and image.height < max_size[1]:
                    counter += 1

    print(counter)


def reorganize_images(directory, object_class, train_part=0.6, val_part=0.25):
    for folder in directory.iterdir():
        image_count = len(list(folder.iterdir()))
        train_cap = np.floor(train_part * image_count)
        validation_cap = np.floor((train_part + val_part) * image_count)

        for counter, image in enumerate(folder.iterdir(), 1):
            destination_dir = (
                TRAIN_DIR / object_class if counter <= train_cap
                else VALIDATION_DIR / object_class if counter <= validation_cap
                else TEST_DIR / object_class
            )

            shutil.move(image, destination_dir)


def reorganize_startbilder(directory, object_class, train_part=0.6, val_part=0.25):
        image_count = len(list(directory.iterdir()))
        train_cap = np.floor(train_part * image_count)
        validation_cap = np.floor((train_part + val_part) * image_count)

        for counter, image in enumerate(directory.rglob("*"), 1):
            destination_dir = (
                START_BILDER_DATASET_DIR / "train" / object_class if counter <= train_cap
                else START_BILDER_DATASET_DIR / "validation" / object_class if counter <= validation_cap
                else START_BILDER_DATASET_DIR / "test" / object_class
            )
            
            shutil.move(image, destination_dir)


def sort_images_to_folders (directory, startbilder=False, train_part=0.6, val_part=0.25):
    for sub_dir in directory.iterdir():
        object_class = sub_dir.name
        
        if startbilder:
            dir_list = [
                START_BILDER_DATASET_DIR / "train" / object_class, 
                START_BILDER_DATASET_DIR / "validation" / object_class, 
                START_BILDER_DATASET_DIR / "test" / object_class
            ]

            for dir in dir_list:
                dir.mkdir(parents=True, exist_ok=True)

            reorganize_startbilder(sub_dir, object_class, train_part)

        else: 
            dir_list = [
                TRAIN_DIR / object_class, 
                VALIDATION_DIR / object_class, 
                TEST_DIR / object_class
            ]

            for dir in dir_list:
                dir.mkdir(parents=True, exist_ok=True)

            reorganize_images(sub_dir, object_class, train_part)


if __name__ == "__main__":
    # resize_directory(REVIEW_DIR)
    # count_missed_images(REVIEW_DIR)
    # sort_images_to_folders(REVIEW_DIR, train_part=1)
    sort_images_to_folders(CANCER_SOURCE_DIR, startbilder=True)
    # sort_images_to_folders(START_BILDER_DIR, startbilder=True)
    pass