from pathlib import Path
import keras

BASE_DIR = Path(__file__).resolve().parent.parent

EXPERIMENTS_DIR = BASE_DIR / "experimental_models"
CASE_OWN_IMAGES_DIR = EXPERIMENTS_DIR / "own_images_aug"
MODELS_DIR = CASE_OWN_IMAGES_DIR / "models"
MODEL_PATH = MODELS_DIR / "cnn_model_just_aug_slow.keras"


def build_model(learning_rate=0.001, output_count=8, regularizer=None, dropout=None, model_type="resnet50"):
    if model_type == "resnet50":
        input_count = 2048
    if model_type == "densenet121":
        input_count = 1024
    
    inputs = keras.Input(shape=(input_count,))

    parameters = {
        "units": output_count,
        "activation": "softmax"
    }

    x = inputs

    if dropout is not None:
        x = keras.layers.Dropout(dropout)(x)

    if regularizer is not None:
        parameters["kernel_regularizer"] = keras.regularizers.l2(regularizer)

    outputs = keras.layers.Dense(**parameters)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_model(destination_path: Path, model, variant):
    destination_path.mkdir(parents=True, exist_ok=True)
    model_path = destination_path / f"cnn_model_{variant}.keras"

    model.save(model_path)


def load_model(model_dir: Path, variant, compile=False):
    model_path = model_dir / f"cnn_model_{variant}.keras" 

    return keras.models.load_model(model_path, compile=compile)


def build_feature_extractor():
    base_model = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))

    x = base_model(inputs, training=False)

    outputs = keras.layers.GlobalAveragePooling2D()(x)

    model = keras.Model(inputs, outputs, name="feature_extractor")

    return model


def split_base_model (model_type="resnet50"):

    if model_type == "resnet50":
        base_model = keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

        cut_layer = base_model.get_layer("conv5_block2_out")

    if model_type == "densenet121":
        base_model = keras.applications.DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

        cut_layer = base_model.get_layer("pool4_pool")
    
    base_model.trainable = False

    feature_extractor = keras.Model(
        inputs=base_model.input,
        outputs=cut_layer.output,
        name="ft_feature_extractor"
    )

    fine_tune_base = keras.Model(
        inputs=cut_layer.output,
        outputs=base_model.output,
        name="fine_tune_backbone"
    )

    return feature_extractor, fine_tune_base


def build_finetuning_model (
    learning_rate=0.00001, 
    pre_trained_head=False, 
    output_count=8, 
    regularizer=None, 
    dropout=None,
    model_type="resnet50",
    batch_trainable=False
):
    
    _, fine_tune_base = split_base_model(model_type=model_type)

    fine_tune_base.trainable = True

    for layer in fine_tune_base.layers:
        if not batch_trainable and isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    inputs = fine_tune_base.input

    x = fine_tune_base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)

    classifier_head = (
        keras.models.load_model(MODEL_PATH) 
        if pre_trained_head 
        else build_model(
            learning_rate=learning_rate, 
            output_count=output_count, 
            regularizer=regularizer,
            dropout=dropout,
            model_type=model_type
        )
    )

    outputs = classifier_head(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model