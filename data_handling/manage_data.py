import tensorflow as tf
import keras
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight


# CLASS_NAMES = [
#     "brille",
#     "flasche",
#     "geldboerse",
#     "handy",
#     "hund",
#     "kirche",
#     "schuh",
#     "tasse"
# ]

CLASS_NAMES = [
    "Benign",
    "Early",
    "Pre",
    "Pro"
]

def preprocess_dataset (dataset, is_cache=True, model_type="resnet50"):
    if model_type == "resnet50":
        preprocess_input = keras.applications.resnet50.preprocess_input
    if model_type == "densenet121":
        preprocess_input = keras.applications.densenet.preprocess_input

    dataset = dataset.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if is_cache:
        dataset = dataset.cache()

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_dataset(
    directory, 
    class_names=CLASS_NAMES, 
    batch_size=32, 
    shuffle=False, 
    is_preprocess=True, 
    model_type="resnet50"
):

    dataset = keras.utils.image_dataset_from_directory(
        directory=directory,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        image_size=(224, 224),
        batch_size=batch_size,
        pad_to_aspect_ratio=True,
        shuffle=shuffle
    )

    if is_preprocess:
        dataset = preprocess_dataset(dataset, model_type)

    return dataset


def extract_features(feature_extractor, dataset):
    labels = np.concatenate([
        labels.numpy()
        for _, labels in dataset
    ])

    images = dataset.map(lambda images, _: images)

    features = feature_extractor.predict(images, verbose=0)

    return features, labels


def load_features (feature_dir: Path, version):
    feature_path = feature_dir / f"features_{str(version)}.npz"

    data = np.load(feature_path)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_validation = data["X_validation"]
    y_validation = data["y_validation"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def get_classweights(y_ds):
    classes = np.unique(y_ds)

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_ds
    )

    return dict(zip(classes, class_weights_array))