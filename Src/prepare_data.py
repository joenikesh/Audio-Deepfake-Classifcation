from pathlib import Path
import shutil
import random

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "raw_data"
OUTPUT_DIR = PROJECT_DIR / "data"

REAL_DIR = OUTPUT_DIR / "real"
FAKE_DIR = OUTPUT_DIR / "fake"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Total real = 3000
MAX_JSUT_REAL_FILES = 1500
MAX_FOR_REAL_FILES = 1500

# Total fake = 3000
MAX_FOR_FAKE_FILES = 1500

# 1500 fake samples from WaveFake
# 1500 / 9 folders ≈ 167 files per folder
MAX_WAVEFAKE_FILES_PER_FOLDER = 167

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def find_audio_files(folder):
    folder = Path(folder)

    if not folder.exists():
        print(f"Missing folder: {folder}")
        return []

    files = [
        file for file in folder.rglob("*")
        if file.suffix.lower() in AUDIO_EXTENSIONS
    ]

    random.shuffle(files)
    return files


def copy_files(files, target_dir, label_prefix, max_files):
    target_dir.mkdir(parents=True, exist_ok=True)

    selected_files = files[:max_files]

    for index, file_path in enumerate(selected_files):
        new_name = f"{label_prefix}_{index:05d}{file_path.suffix.lower()}"
        destination = target_dir / new_name
        shutil.copy2(file_path, destination)

    print(f"Copied {len(selected_files)} files to {target_dir}")


def prepare_jsut_real():
    jsut_dir = RAW_DIR / "jsut"

    files = find_audio_files(jsut_dir)

    copy_files(
        files=files,
        target_dir=REAL_DIR,
        label_prefix="jsut_real",
        max_files=MAX_JSUT_REAL_FILES
    )


def prepare_wavefake_fake():
    wavefake_dir = RAW_DIR / "wavefake"

    fake_folders = [
    "jsut_multi_band_melgan",
    "jsut_parallel_wavegan",
    "ljspeech_full_band_melgan",
    "ljspeech_hifiGAN",
    "ljspeech_melgan",
    "ljspeech_melgan_large",
    "ljspeech_multi_band_melgan",
    "ljspeech_parallel_wavegan",
    "ljspeech_waveglow",
]

    for folder_name in fake_folders:
        folder_path = wavefake_dir / folder_name
        files = find_audio_files(folder_path)

        copy_files(
            files=files,
            target_dir=FAKE_DIR,
            label_prefix=f"wavefake_{folder_name}",
            max_files=MAX_WAVEFAKE_FILES_PER_FOLDER
        )


def prepare_fake_or_real():
    for_original_dir = (
        RAW_DIR
        / "fake_or_real"
        / "for-original"
        / "for-original"
    )

    real_folders = [
        for_original_dir / "training" / "real",
        for_original_dir / "testing" / "real",
        for_original_dir / "validation" / "real",
    ]

    fake_folders = [
        for_original_dir / "training" / "fake",
        for_original_dir / "testing" / "fake",
        for_original_dir / "validation" / "fake",
    ]

    real_files = []
    fake_files = []

    for folder in real_folders:
        real_files.extend(find_audio_files(folder))

    for folder in fake_folders:
        fake_files.extend(find_audio_files(folder))

    random.shuffle(real_files)
    random.shuffle(fake_files)

    copy_files(
        files=real_files,
        target_dir=REAL_DIR,
        label_prefix="for_real",
        max_files=MAX_FOR_REAL_FILES
    )

    copy_files(
        files=fake_files,
        target_dir=FAKE_DIR,
        label_prefix="for_fake",
        max_files=MAX_FOR_FAKE_FILES
    )

def main():
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    prepare_jsut_real()
    prepare_wavefake_fake()
    prepare_fake_or_real()

    print("\nDataset preparation complete.")
    print(f"Real files: {len(list(REAL_DIR.glob('*')))}")
    print(f"Fake files: {len(list(FAKE_DIR.glob('*')))}")


if __name__ == "__main__":
    main()