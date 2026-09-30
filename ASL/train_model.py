"""Train ISL alphabet and common-phrase models from the bundled datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
ALPHABET_DIR = ROOT / "DATABASE ISL" / "ISL_Dataset"
PHRASE_ZIP = ROOT / "Dataset of Common phrases in ISL" / "Dataset of Common phrases in Indian Sign Language.zip"
GENERATED_DIR = ROOT / "generated" / "phrase_dataset"
MODEL_DIR = ROOT / "models"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{description} was not found at {path}. "
            "Check the dataset paths in TRAINING.md."
        )


def train_alphabet(epochs: int) -> dict[str, object]:
    require_directory(ALPHABET_DIR, "Alphabet dataset")
    labels = sorted(path.name for path in ALPHABET_DIR.iterdir() if path.is_dir())
    if not labels:
        raise RuntimeError(f"No alphabet label folders found in {ALPHABET_DIR}")
    train_ds, validation_ds = tf.keras.utils.image_dataset_from_directory(
        ALPHABET_DIR, labels="inferred", label_mode="int", class_names=labels,
        image_size=(128, 128), batch_size=32, validation_split=0.2, subset="both", seed=42,
    )
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.15),
    ])
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)), augmentation,
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"), tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(len(labels), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "isl_alphabet.keras")
    write_json(MODEL_DIR / "isl_alphabet_labels.json", labels)
    return {"labels": labels, "final_validation_accuracy": history.history["val_accuracy"][-1]}


def extract_phrase_dataset() -> None:
    if GENERATED_DIR.exists():
        return
    if not PHRASE_ZIP.is_file():
        raise FileNotFoundError(
            f"Phrase dataset archive was not found at {PHRASE_ZIP}. "
            "Check the dataset paths in TRAINING.md."
        )
    GENERATED_DIR.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PHRASE_ZIP) as archive:
        destination = GENERATED_DIR.resolve()
        for member in archive.infolist():
            member_path = (GENERATED_DIR / member.filename).resolve()
            if destination not in member_path.parents:
                raise ValueError(f"Unsafe path in phrase dataset archive: {member.filename}")
        archive.extractall(GENERATED_DIR)


def load_phrase_sequences() -> tuple[np.ndarray, np.ndarray, list[str]]:
    extract_phrase_dataset()
    data_root = GENERATED_DIR / "MP_Data"
    require_directory(data_root, "Extracted phrase dataset")
    labels = sorted(path.name for path in data_root.iterdir() if path.is_dir())
    samples: list[np.ndarray] = []
    targets: list[int] = []
    feature_size: int | None = None
    for target, label in enumerate(labels):
        sequence_dirs = [path for path in (data_root / label).iterdir() if path.is_dir() and path.name.isdigit()]
        for sequence_dir in sorted(sequence_dirs, key=lambda path: int(path.name)):
            frame_paths = sorted(sequence_dir.glob("*.npy"), key=lambda path: int(path.stem))
            if len(frame_paths) != 30:
                raise ValueError(f"Expected 30 frames in {sequence_dir}, found {len(frame_paths)}")
            frames = [np.asarray(np.load(path), dtype=np.float32).reshape(-1) for path in frame_paths]
            if feature_size is None:
                feature_size = frames[0].size
            if any(frame.size != feature_size for frame in frames):
                sizes = sorted({frame.size for frame in frames})
                raise ValueError(f"Expected {feature_size} landmark values in {sequence_dir}, found {sizes}")
            samples.append(np.stack(frames))
            targets.append(target)
    if not samples:
        raise RuntimeError(f"No phrase sequences found in {data_root}")
    return np.stack(samples), np.asarray(targets), labels


def train_phrases(epochs: int) -> dict[str, object]:
    samples, targets, labels = load_phrase_sequences()
    train_x, validation_x, train_y, validation_y = train_test_split(
        samples, targets, test_size=0.2, random_state=42, stratify=targets
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30, samples.shape[2])), tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.LSTM(64, return_sequences=True), tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(labels), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_x, train_y, validation_data=(validation_x, validation_y), epochs=epochs, batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "isl_phrases.keras")
    write_json(MODEL_DIR / "isl_phrase_labels.json", labels)
    return {"samples": int(len(samples)), "labels": labels, "final_validation_accuracy": history.history["val_accuracy"][-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs for each model")
    parser.add_argument("--clean", action="store_true", help="Re-extract the phrase ZIP before training")
    parser.add_argument("--only", choices=("alphabet", "phrases", "both"), default="both", help="Choose which model to train")
    args = parser.parse_args()
    if args.clean and GENERATED_DIR.exists():
        shutil.rmtree(GENERATED_DIR)
    tf.random.set_seed(42)
    results = {}
    if args.only in ("alphabet", "both"):
        results["alphabet"] = train_alphabet(args.epochs)
    if args.only in ("phrases", "both"):
        results["phrases"] = train_phrases(args.epochs)
    metrics_path = MODEL_DIR / "training_metrics.json"
    previous = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    previous.update(results)
    write_json(metrics_path, previous)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()