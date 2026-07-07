from pathlib import Path
import numpy as np
import librosa



PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
REAL_DIR = DATA_DIR / "real"
FAKE_DIR = DATA_DIR / "fake"
OUTPUT_DIR = DATA_DIR / "processed"




SAMPLE_RATE = 16000
DURATION = 8
N_MFCC = 40

MAX_SAMPLES = SAMPLE_RATE * DURATION




def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio


def fix_audio_length(audio):
    if len(audio) < MAX_SAMPLES:
        padding = MAX_SAMPLES - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant")
    else:
        audio = audio[:MAX_SAMPLES]

    return audio


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    return mfcc


def preprocess_file(file_path):
    audio = load_audio(file_path)
    audio = fix_audio_length(audio)
    mfcc = extract_mfcc(audio)

    return mfcc



def preprocess_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X = []
    y = []

    label_map = {
        "real": 0,
        "fake": 1
    }

    for label_name, label_value in label_map.items():
        folder = DATA_DIR / label_name

        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue

        audio_files = []
        audio_files.extend(folder.glob("*.wav"))
        audio_files.extend(folder.glob("*.mp3"))
        audio_files.extend(folder.glob("*.flac"))
        audio_files.extend(folder.glob("*.m4a"))

        print(f"Processing {len(audio_files)} {label_name} files")

        for file_path in audio_files:
            try:
                mfcc = preprocess_file(file_path)
                X.append(mfcc)
                y.append(label_value)

            except Exception as e:
                print(f"Could not process {file_path.name}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)

    print("\nPreprocessing complete.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Saved X to: {OUTPUT_DIR / 'X.npy'}")
    print(f"Saved y to: {OUTPUT_DIR / 'y.npy'}")


if __name__ == "__main__":
    preprocess_dataset()