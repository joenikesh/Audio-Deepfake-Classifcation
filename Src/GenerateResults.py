from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

METRICS_FILE = RESULTS_DIR / "training_metrics.json"
LOSS_GRAPH_FILE = RESULTS_DIR / "loss_graph.png"
ACCURACY_GRAPH_FILE = RESULTS_DIR / "accuracy_graph.png"
CONFUSION_MATRIX_FILE = RESULTS_DIR / "confusion_matrix.png"
TEST_METRICS_FILE = RESULTS_DIR / "test_metrics_graph.png"
SUMMARY_FILE = RESULTS_DIR / "evaluation_summary.txt"


def load_metrics() -> dict:
    """
    Load the metrics created by Train.py.

    The training script stores the epoch history, dataset information,
    training configuration, and final test results in one JSON file.
    """

    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {METRICS_FILE}.\n"
            "Run Src/Train.py before generating evaluation graphs."
        )

    with METRICS_FILE.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    required_sections = {
        "training_history",
        "test_metrics",
        "dataset",
        "configuration",
    }

    missing_sections = required_sections.difference(metrics)

    if missing_sections:
        raise KeyError(
            "The metrics file is missing these sections: "
            f"{sorted(missing_sections)}"
        )

    if not metrics["training_history"]:
        raise ValueError("The training history is empty.")

    return metrics


def create_loss_graph(training_history: list[dict]) -> None:
    """
    Plot training and validation loss for every completed epoch.

    A decreasing loss indicates that the model is improving. A large gap
    between training and validation loss may indicate overfitting.
    """

    epochs = [record["epoch"] for record in training_history]
    training_loss = [record["train_loss"] for record in training_history]
    validation_loss = [
        record["validation_loss"] for record in training_history
    ]

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, training_loss, marker="o", label="Training loss")
    plt.plot(
        epochs,
        validation_loss,
        marker="o",
        label="Validation loss",
    )

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_GRAPH_FILE, dpi=300)
    plt.close()


def create_accuracy_graph(training_history: list[dict]) -> None:
    """
    Plot training and validation accuracy across epochs.

    This graph shows how classification performance changed during
    training and whether validation performance followed training.
    """

    epochs = [record["epoch"] for record in training_history]
    training_accuracy = [
        record["train_accuracy"] * 100
        for record in training_history
    ]
    validation_accuracy = [
        record["validation_accuracy"] * 100
        for record in training_history
    ]

    plt.figure(figsize=(9, 6))
    plt.plot(
        epochs,
        training_accuracy,
        marker="o",
        label="Training accuracy",
    )
    plt.plot(
        epochs,
        validation_accuracy,
        marker="o",
        label="Validation accuracy",
    )

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ACCURACY_GRAPH_FILE, dpi=300)
    plt.close()


def create_confusion_matrix_graph(confusion_matrix: list[list[int]]) -> None:
    """
    Visualize correct and incorrect predictions for real and fake audio.

    Rows represent the actual class and columns represent the class
    predicted by the model.
    """

    matrix = np.asarray(confusion_matrix, dtype=int)

    if matrix.shape != (2, 2):
        raise ValueError(
            "Expected a 2 by 2 confusion matrix for binary classification. "
            f"Received shape {matrix.shape}."
        )

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix)

    figure.colorbar(image, ax=axis)

    class_names = ["Real", "Fake"]
    axis.set_xticks(range(2), labels=class_names)
    axis.set_yticks(range(2), labels=class_names)

    axis.set_title("Test Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")

    threshold = matrix.max() / 2

    for row in range(2):
        for column in range(2):
            text_color = "white" if matrix[row, column] > threshold else "black"

            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=14,
            )

    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_FILE, dpi=300)
    plt.close(figure)


def create_test_metrics_graph(test_metrics: dict) -> None:
    """
    Compare the four primary classification metrics on the test set.

    Accuracy measures overall correctness. Precision measures how often a
    fake prediction is correct. Recall measures how many fake files are
    detected. F1 balances precision and recall.
    """

    metric_names = ["Accuracy", "Precision", "Recall", "F1 score"]
    metric_values = [
        test_metrics["accuracy"] * 100,
        test_metrics["precision"] * 100,
        test_metrics["recall"] * 100,
        test_metrics["f1_score"] * 100,
    ]

    figure, axis = plt.subplots(figsize=(8, 6))
    bars = axis.bar(metric_names, metric_values)

    axis.set_title("Test Classification Metrics")
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, 100)

    for bar, value in zip(bars, metric_values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()
    figure.savefig(TEST_METRICS_FILE, dpi=300)
    plt.close(figure)


def create_summary(metrics: dict) -> None:
    """
    Save a readable text summary for reports, presentations, and GitHub.

    The summary includes the best epoch, dataset split, model performance,
    confusion matrix, and the full classification report.
    """

    dataset = metrics["dataset"]
    test_metrics = metrics["test_metrics"]
    configuration = metrics["configuration"]
    best_epoch = metrics["best_epoch"]

    confusion_matrix = test_metrics["confusion_matrix"]
    true_real = confusion_matrix[0][0]
    real_predicted_fake = confusion_matrix[0][1]
    fake_predicted_real = confusion_matrix[1][0]
    true_fake = confusion_matrix[1][1]

    summary = f"""AMAD Model Evaluation Summary
=============================

Dataset
-------
Total samples: {dataset["total_samples"]}
Training samples: {dataset["training_samples"]}
Validation samples: {dataset["validation_samples"]}
Test samples: {dataset["test_samples"]}

Feature shapes
--------------
MFCC: {dataset["mfcc_shape"]}
Pitch: {dataset["pitch_shape"]}
Energy: {dataset["energy_shape"]}

Training configuration
----------------------
Device: {configuration["device"]}
Batch size: {configuration["batch_size"]}
Initial learning rate: {configuration["learning_rate"]}
Maximum epochs: {configuration["maximum_epochs"]}
Best epoch: {best_epoch}

Test metrics
------------
Loss: {test_metrics["loss"]:.4f}
Accuracy: {test_metrics["accuracy"]:.4f} ({test_metrics["accuracy"] * 100:.2f}%)
Precision: {test_metrics["precision"]:.4f} ({test_metrics["precision"] * 100:.2f}%)
Recall: {test_metrics["recall"]:.4f} ({test_metrics["recall"] * 100:.2f}%)
F1 score: {test_metrics["f1_score"]:.4f} ({test_metrics["f1_score"] * 100:.2f}%)

Confusion matrix interpretation
-------------------------------
Real correctly classified as real: {true_real}
Real incorrectly classified as fake: {real_predicted_fake}
Fake incorrectly classified as real: {fake_predicted_real}
Fake correctly classified as fake: {true_fake}

Classification report
---------------------
{test_metrics["classification_report"]}
"""

    SUMMARY_FILE.write_text(summary, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()
    training_history = metrics["training_history"]
    test_metrics = metrics["test_metrics"]

    create_loss_graph(training_history)
    create_accuracy_graph(training_history)
    create_confusion_matrix_graph(test_metrics["confusion_matrix"])
    create_test_metrics_graph(test_metrics)
    create_summary(metrics)

    print("Evaluation files generated successfully:")
    print(f"- {LOSS_GRAPH_FILE}")
    print(f"- {ACCURACY_GRAPH_FILE}")
    print(f"- {CONFUSION_MATRIX_FILE}")
    print(f"- {TEST_METRICS_FILE}")
    print(f"- {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
