import shutil
from pathlib import Path


def export_misclassified_images(confusion_df, destination_dir, class_names):
    destination_dir = Path(destination_dir)

    for class_name in class_names:
        (destination_dir / class_name).mkdir(parents=True, exist_ok=True)

    confusion_df["y_proba"] = (confusion_df["y_proba"] * 100).round().astype(int)
    confusion_df["num"] = (confusion_df.groupby(["y_true", "y_pred"]).cumcount() + 1)

    for _, row in confusion_df.iterrows():
        source = Path(row["image_path"])

        image_name = f"{row['y_pred']}_{row['y_proba']}_{row['num']}{source.suffix}"

        destination = (destination_dir / row["y_true"] / image_name)

        shutil.copy2(source, destination)