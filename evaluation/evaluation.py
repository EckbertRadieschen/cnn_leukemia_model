import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss
)

import pandas as pd

# CLASS_NAMES = [
#     "tasse",
#     "geldboerse",
#     "flasche",
#     "brille",
#     "schuh",
#     "handy",
#     "hund",
#     "kirche"
# ]

CLASS_NAMES = [
    "Benign",
    "Early",
    "Pre",
    "Pro"
]


def evaluate_model(model, X, y_true, class_names: list[str]):
    probabilities = model.predict(X)

    y_pred = np.argmax(probabilities, axis=1)

    y_proba = [np.max(prob) for prob in probabilities]

    pred_data = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, probabilities, labels=range(len(class_names))),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(class_names))),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=range(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
    }

    results_df = pd.DataFrame(pred_data)

    results_df["y_true"] = results_df["y_true"].apply(lambda x: class_names[x])
    results_df["y_pred"] = results_df["y_pred"].apply(lambda x: class_names[x])

    return results, results_df


