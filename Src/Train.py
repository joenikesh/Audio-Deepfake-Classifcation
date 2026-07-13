from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset




PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)




RANDOM_SEED = 42

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

NUM_WORKERS = 0

MODEL_PATH = MODEL_DIR / "amad_best_model.pt"
METRICS_PATH = RESULTS_DIR / "training_metrics.json"



def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU through MPS")

    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device




def load_processed_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    required_files = {
        "X_mfcc": PROCESSED_DIR / "X_mfcc.npy",
        "X_pitch": PROCESSED_DIR / "X_pitch.npy",
        "X_energy": PROCESSED_DIR / "X_energy.npy",
        "y": PROCESSED_DIR / "y.npy",
    }

    missing_files = [
        str(path)
        for path in required_files.values()
        if not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(missing_files)

        raise FileNotFoundError(
            "The following preprocessing output files are missing:\n"
            f"{missing_text}\n\n"
            "Run preprocessing before model training."
        )

    X_mfcc = np.load(required_files["X_mfcc"])
    X_pitch = np.load(required_files["X_pitch"])
    X_energy = np.load(required_files["X_energy"])
    y = np.load(required_files["y"])

    validate_data_shapes(X_mfcc, X_pitch, X_energy, y)

    print("\nLoaded preprocessed data")
    print(f"X_mfcc shape:   {X_mfcc.shape}")
    print(f"X_pitch shape:  {X_pitch.shape}")
    print(f"X_energy shape: {X_energy.shape}")
    print(f"y shape:        {y.shape}")

    unique_labels, label_counts = np.unique(y, return_counts=True)

    print("\nClass distribution")

    for label, count in zip(unique_labels, label_counts):
        class_name = "real" if int(label) == 0 else "fake"
        print(f"Label {label} ({class_name}): {count}")

    return X_mfcc, X_pitch, X_energy, y


def validate_data_shapes(
    X_mfcc: np.ndarray,
    X_pitch: np.ndarray,
    X_energy: np.ndarray,
    y: np.ndarray,
) -> None:
    number_of_samples = len(y)

    if X_mfcc.ndim != 3:
        raise ValueError(
            "X_mfcc must have shape "
            "(samples, mfcc_features, time_frames). "
            f"Received {X_mfcc.shape}."
        )

    if X_pitch.ndim != 2:
        raise ValueError(
            "X_pitch must have shape (samples, time_frames). "
            f"Received {X_pitch.shape}."
        )

    if X_energy.ndim != 2:
        raise ValueError(
            "X_energy must have shape (samples, time_frames). "
            f"Received {X_energy.shape}."
        )

    if y.ndim != 1:
        y = y.reshape(-1)

    if len(X_mfcc) != number_of_samples:
        raise ValueError("X_mfcc and y have different sample counts.")

    if len(X_pitch) != number_of_samples:
        raise ValueError("X_pitch and y have different sample counts.")

    if len(X_energy) != number_of_samples:
        raise ValueError("X_energy and y have different sample counts.")

    number_of_frames = X_mfcc.shape[2]

    if X_pitch.shape[1] != number_of_frames:
        raise ValueError(
            "Pitch and MFCC features have different frame counts: "
            f"{X_pitch.shape[1]} and {number_of_frames}."
        )

    if X_energy.shape[1] != number_of_frames:
        raise ValueError(
            "Energy and MFCC features have different frame counts: "
            f"{X_energy.shape[1]} and {number_of_frames}."
        )

    unique_labels = set(np.unique(y).tolist())

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            "This model expects binary labels containing only 0 and 1. "
            f"Found labels: {sorted(unique_labels)}"
        )




def create_data_splits(
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(len(y))

    train_validation_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    remaining_validation_fraction = (
        VALIDATION_SIZE / (1.0 - TEST_SIZE)
    )

    train_indices, validation_indices = train_test_split(
        train_validation_indices,
        test_size=remaining_validation_fraction,
        random_state=RANDOM_SEED,
        stratify=y[train_validation_indices],
    )

    print("\nDataset split")
    print(f"Training samples:   {len(train_indices)}")
    print(f"Validation samples: {len(validation_indices)}")
    print(f"Test samples:       {len(test_indices)}")

    return train_indices, validation_indices, test_indices



class FeatureNormalizer:
    def __init__(self) -> None:
        self.mfcc_mean: np.ndarray | None = None
        self.mfcc_std: np.ndarray | None = None

        self.pitch_mean: float | None = None
        self.pitch_std: float | None = None

        self.energy_mean: float | None = None
        self.energy_std: float | None = None

    def fit(
        self,
        X_mfcc_train: np.ndarray,
        X_pitch_train: np.ndarray,
        X_energy_train: np.ndarray,
    ) -> None:
        # One mean and standard deviation for each MFCC coefficient.
        self.mfcc_mean = X_mfcc_train.mean(
            axis=(0, 2),
            keepdims=True,
        )

        self.mfcc_std = X_mfcc_train.std(
            axis=(0, 2),
            keepdims=True,
        )

        self.mfcc_std = np.maximum(self.mfcc_std, 1e-8)

        self.pitch_mean = float(X_pitch_train.mean())
        self.pitch_std = max(float(X_pitch_train.std()), 1e-8)

        self.energy_mean = float(X_energy_train.mean())
        self.energy_std = max(float(X_energy_train.std()), 1e-8)

    def transform(
        self,
        X_mfcc: np.ndarray,
        X_pitch: np.ndarray,
        X_energy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.mfcc_mean is None or self.mfcc_std is None:
            raise RuntimeError("The normalizer must be fitted first.")

        if self.pitch_mean is None or self.pitch_std is None:
            raise RuntimeError("Pitch normalization values are missing.")

        if self.energy_mean is None or self.energy_std is None:
            raise RuntimeError("Energy normalization values are missing.")

        normalized_mfcc = (
            (X_mfcc - self.mfcc_mean) / self.mfcc_std
        ).astype(np.float32)

        normalized_pitch = (
            (X_pitch - self.pitch_mean) / self.pitch_std
        ).astype(np.float32)

        normalized_energy = (
            (X_energy - self.energy_mean) / self.energy_std
        ).astype(np.float32)

        return (
            normalized_mfcc,
            normalized_pitch,
            normalized_energy,
        )

    def state_dict(self) -> dict[str, np.ndarray | float]:
        if self.mfcc_mean is None or self.mfcc_std is None:
            raise RuntimeError("The normalizer has not been fitted.")

        return {
            "mfcc_mean": self.mfcc_mean,
            "mfcc_std": self.mfcc_std,
            "pitch_mean": self.pitch_mean,
            "pitch_std": self.pitch_std,
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
        }




class AudioFeatureDataset(Dataset):
    def __init__(
        self,
        X_mfcc: np.ndarray,
        X_pitch: np.ndarray,
        X_energy: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.X_mfcc = torch.from_numpy(
            X_mfcc[indices].astype(np.float32)
        )

        self.X_pitch = torch.from_numpy(
            X_pitch[indices].astype(np.float32)
        )

        self.X_energy = torch.from_numpy(
            X_energy[indices].astype(np.float32)
        )

        self.y = torch.from_numpy(
            y[indices].astype(np.float32)
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X_mfcc[index],
            self.X_pitch[index],
            self.X_energy[index],
            self.y[index],
        )




class AMADModel(nn.Module):
    """
    MultiLingual Audio Authenticity Detector.

    Input shapes:
        MFCC:   (batch, 40, 251)
        pitch:  (batch, 251)
        energy: (batch, 251)
    """

    def __init__(
        self,
        lstm_hidden_size: int = 128,
        dropout_probability: float = 0.30,
    ) -> None:
        super().__init__()

        # MFCC is treated like a one-channel image:
        # height = MFCC coefficients
        # width = time frames
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(dropout_probability),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(dropout_probability),
        )

        # Reduce the MFCC-frequency dimension to one while preserving time.
        self.frequency_pool = nn.AdaptiveAvgPool2d((1, None))

        # 64 CNN features + pitch + energy.
        lstm_input_size = 64 + 2

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        mfcc: torch.Tensor,
        pitch: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        # Before:
        # (batch, 40, 251)
        #
        # After:
        # (batch, 1, 40, 251)
        mfcc = mfcc.unsqueeze(1)

        # CNN output:
        # (batch, 64, reduced_mfcc_dimension, 251)
        cnn_output = self.cnn(mfcc)

        # Pool only across the MFCC dimension.
        # Result:
        # (batch, 64, 1, 251)
        cnn_output = self.frequency_pool(cnn_output)

        # Remove the dimension of size one.
        # Result:
        # (batch, 64, 251)
        cnn_output = cnn_output.squeeze(2)

        # Convert to:
        # (batch, 251, 64)
        cnn_output = cnn_output.transpose(1, 2)

        # Convert pitch and energy from:
        # (batch, 251)
        #
        # To:
        # (batch, 251, 1)
        pitch = pitch.unsqueeze(-1)
        energy = energy.unsqueeze(-1)

        # Combine all features for every time frame:
        # (batch, 251, 66)
        combined_features = torch.cat(
            [cnn_output, pitch, energy],
            dim=2,
        )

        # LSTM output:
        # (batch, 251, lstm_hidden_size * 2)
        lstm_output, _ = self.lstm(combined_features)

        # Global average pooling across all time frames:
        # (batch, lstm_hidden_size * 2)
        pooled_output = lstm_output.mean(dim=1)

        # One raw output score per audio file.
        logits = self.classifier(pooled_output)

        return logits.squeeze(1)



def create_data_loaders(
    X_mfcc: np.ndarray,
    X_pitch: np.ndarray,
    X_energy: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = AudioFeatureDataset(
        X_mfcc,
        X_pitch,
        X_energy,
        y,
        train_indices,
    )

    validation_dataset = AudioFeatureDataset(
        X_mfcc,
        X_pitch,
        X_energy,
        y,
        validation_indices,
    )

    test_dataset = AudioFeatureDataset(
        X_mfcc,
        X_pitch,
        X_energy,
        y,
        test_indices,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_loader, validation_loader, test_loader




def run_training_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    for mfcc, pitch, energy, labels in data_loader:
        mfcc = mfcc.to(device)
        pitch = pitch.to(device)
        energy = energy.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(mfcc, pitch, energy)
        loss = loss_function(logits, labels)

        loss.backward()

        # Helps prevent unstable LSTM gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long()

        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_labels.extend(labels.long().cpu().numpy().tolist())

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()

    total_loss = 0.0

    all_probabilities: list[float] = []
    all_predictions: list[int] = []
    all_labels: list[int] = []

    for mfcc, pitch, energy, labels in data_loader:
        mfcc = mfcc.to(device)
        pitch = pitch.to(device)
        energy = energy.to(device)
        labels = labels.to(device)

        logits = model(mfcc, pitch, energy)
        loss = loss_function(logits, labels)

        total_loss += loss.item() * labels.size(0)

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long()

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_labels.extend(
            labels.long().cpu().numpy().tolist()
        )

    average_loss = total_loss / len(data_loader.dataset)

    metrics = {
        "loss": average_loss,
        "accuracy": accuracy_score(
            all_labels,
            all_predictions,
        ),
        "precision": precision_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            all_labels,
            all_predictions,
        ).tolist(),
        "classification_report": classification_report(
            all_labels,
            all_predictions,
            target_names=["real", "fake"],
            zero_division=0,
        ),
        "labels": all_labels,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
    }

    return metrics



def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: dict[str, object],
    normalizer: FeatureNormalizer,
    number_of_mfcc: int,
    number_of_frames: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": validation_metrics,
        "normalizer": normalizer.state_dict(),
        "number_of_mfcc": number_of_mfcc,
        "number_of_frames": number_of_frames,
        "label_mapping": {
            0: "real",
            1: "fake",
        },
    }

    torch.save(checkpoint, MODEL_PATH)



def main() -> None:
    set_random_seed(RANDOM_SEED)

    device = get_device()

    X_mfcc, X_pitch, X_energy, y = load_processed_data()

    # Ensure labels are one-dimensional.
    y = y.reshape(-1).astype(np.int64)

    train_indices, validation_indices, test_indices = (
        create_data_splits(y)
    )

    # Fit normalization only on training data.
    # This prevents information from the validation and test sets
    # leaking into model training.
    normalizer = FeatureNormalizer()

    normalizer.fit(
        X_mfcc[train_indices],
        X_pitch[train_indices],
        X_energy[train_indices],
    )

    X_mfcc, X_pitch, X_energy = normalizer.transform(
        X_mfcc,
        X_pitch,
        X_energy,
    )

    train_loader, validation_loader, test_loader = (
        create_data_loaders(
            X_mfcc,
            X_pitch,
            X_energy,
            y,
            train_indices,
            validation_indices,
            test_indices,
            device,
        )
    )

    model = AMADModel(
        lstm_hidden_size=128,
        dropout_probability=0.30,
    ).to(device)

    print("\nModel architecture")
    print(model)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\nTrainable parameters: {trainable_parameters:,}")

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    history: list[dict[str, float | int]] = []

    print("\nStarting model training\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_accuracy = run_training_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )

        validation_metrics = evaluate_model(
            model,
            validation_loader,
            loss_function,
            device,
        )

        validation_loss = float(validation_metrics["loss"])
        validation_accuracy = float(
            validation_metrics["accuracy"]
        )
        validation_f1 = float(validation_metrics["f1_score"])

        scheduler.step(validation_loss)

        current_learning_rate = optimizer.param_groups[0]["lr"]

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_f1_score": validation_f1,
            "learning_rate": current_learning_rate,
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.4f} | "
            f"validation F1: {validation_f1:.4f} | "
            f"learning rate: {current_learning_rate:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_metrics=validation_metrics,
                normalizer=normalizer,
                number_of_mfcc=X_mfcc.shape[1],
                number_of_frames=X_mfcc.shape[2],
            )

            print(f"Saved improved model to {MODEL_PATH}")

        else:
            epochs_without_improvement += 1

            print(
                "Validation loss did not improve. "
                f"Early stopping count: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping activated.")
            break

    # Load the best model rather than evaluating the final epoch.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(
        model,
        test_loader,
        loss_function,
        device,
    )

    print("\nTest results")
    print(f"Loss:      {test_metrics['loss']:.4f}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 score:  {test_metrics['f1_score']:.4f}")

    print("\nConfusion matrix")
    print(np.array(test_metrics["confusion_matrix"]))

    print("\nClassification report")
    print(test_metrics["classification_report"])

    results_to_save = {
        "configuration": {
            "random_seed": RANDOM_SEED,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "maximum_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "test_size": TEST_SIZE,
            "validation_size": VALIDATION_SIZE,
            "device": str(device),
        },
        "dataset": {
            "total_samples": len(y),
            "training_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "test_samples": len(test_indices),
            "mfcc_shape": list(X_mfcc.shape),
            "pitch_shape": list(X_pitch.shape),
            "energy_shape": list(X_energy.shape),
        },
        "training_history": history,
        "best_epoch": checkpoint["epoch"],
        "test_metrics": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "classification_report": test_metrics[
                "classification_report"
            ],
        },
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results_to_save, file, indent=4)

    print(f"\nBest model saved to: {MODEL_PATH}")
    print(f"Training results saved to: {METRICS_PATH}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()