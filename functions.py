from io import BytesIO
from PIL import Image

from pathlib import Path

import keras
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

PICTURE_DATA_DIR = BASE_DIR / "picture_data"

START_PICTURES_DIR = PICTURE_DATA_DIR / "start_pictures"


def vectorize_image(image):
    image = image.resize((224, 224))

    image_array = keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    return keras.applications.resnet50.preprocess_input(image_array)


def predict_picture (model, image_array):
    predictions = model.predict(image_array)

    return keras.applications.resnet50.decode_predictions(
        predictions,
        top=3
    )
