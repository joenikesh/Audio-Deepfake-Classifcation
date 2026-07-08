from pathlib import Path
import numpy as np
import librosa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed"

SAMPLE_RATE = 16000
DURATION = 8
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

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
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    return mfcc


def extract_pitch(audio):
    pitches, magnitudes = librosa.piptrack(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    pitch_values = []

    for frame in range(pitches.shape[1]):
        index = magnitudes[:, frame].argmax()
        pitch = pitches[index, frame]
        pitch_values.append(pitch)

    return np.array(pitch_values)


def extract_energy(audio):
    energy = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    return energy[0]


def preprocess_file(file_path):
    audio = load_audio(file_path)
    audio = fix_audio_length(audio)

    mfcc = extract_mfcc(audio)
    pitch = extract_pitch(audio)
    energy = extract_energy(audio)

    return mfcc, pitch, energy


def preprocess_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_mfcc = []
    X_pitch = []
    X_energy = []
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
                mfcc, pitch, energy = preprocess_file(file_path)

                X_mfcc.append(mfcc)
                X_pitch.append(pitch)
                X_energy.append(energy)
                y.append(label_value)

            except Exception as e:
                print(f"Could not process {file_path.name}: {e}")

    X_mfcc = np.array(X_mfcc, dtype=np.float32)
    X_pitch = np.array(X_pitch, dtype=np.float32)
    X_energy = np.array(X_energy, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(OUTPUT_DIR / "X_mfcc.npy", X_mfcc)
    np.save(OUTPUT_DIR / "X_pitch.npy", X_pitch)
    np.save(OUTPUT_DIR / "X_energy.npy", X_energy)
    np.save(OUTPUT_DIR / "y.npy", y)

    print("\nPreprocessing complete.")
    print(f"X_mfcc shape: {X_mfcc.shape}")
    print(f"X_pitch shape: {X_pitch.shape}")
    print(f"X_energy shape: {X_energy.shape}")
    print(f"y shape: {y.shape}")


if __name__ == "__main__":
    preprocess_dataset()