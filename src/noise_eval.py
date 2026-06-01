import os
import sys
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import DATASET_PATH, MODEL_PATH, LABELS_PATH, TEST_SPEAKERS
from core.preprocess import load_audio, extract_embedding

SNR_LEVELS_DB = [30, 20, 15, 10, 5, 0, -5]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return np.clip(audio + noise, -1.0, 1.0)


def get_speaker(filepath: str) -> str:
    stem = re.sub(r'_orig$', '', Path(filepath).stem)
    return 'tts' if stem.startswith('tts_') else stem.lower()


def collect_test_files(dataset_path: str):
    files, labels = [], []
    for class_dir in sorted(Path(dataset_path).iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.glob('*'):
            if f.suffix.lower() not in ('.wav', '.ogg', '.mp3', '.flac'):
                continue
            if get_speaker(str(f)) in TEST_SPEAKERS:
                files.append(str(f))
                labels.append(class_dir.name)
    return files, labels


def evaluate_at_snr(model, class_names: list, files: list,
                    labels: list, snr_db: float | None) -> float:
    correct = 0
    for path, label in zip(files, labels):
        audio = load_audio(path)
        if snr_db is not None:
            audio = add_noise(audio, snr_db)
        emb = extract_embedding(audio)
        probs = model(emb[np.newaxis], training=False).numpy()[0]
        pred = class_names[int(probs.argmax())]
        if pred == label:
            correct += 1
    return correct / len(files) if files else 0.0


def main():
    print('Loading model and data...')
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(LABELS_PATH, allow_pickle=True).tolist()

    f_test, l_test = collect_test_files(DATASET_PATH)
    print(f'Test files (speakers {TEST_SPEAKERS}): {len(f_test)}\n')

    results = []

    print('Baseline (no noise)...', end=' ', flush=True)
    acc_clean = evaluate_at_snr(model, class_names, f_test, l_test, None)
    results.append({'snr_db': 'clean', 'accuracy': acc_clean})
    print(f'{acc_clean*100:.1f}%')

    for snr in SNR_LEVELS_DB:
        print(f'SNR = {snr:>4} dB...', end=' ', flush=True)
        acc = evaluate_at_snr(model, class_names, f_test, l_test, snr)
        results.append({'snr_db': snr, 'accuracy': acc})
        print(f'{acc*100:.1f}%')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'noise_robustness.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['snr_db', 'accuracy'])
        w.writeheader()
        w.writerows(results)
    print(f'\nResults saved: {csv_path}')

    acc_vals = [r['accuracy'] * 100 for r in results]
    x_labels = ['Clean'] + [f'{s} dB' for s in SNR_LEVELS_DB]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    ax.plot(range(len(acc_vals)), acc_vals,
            color='#1f77b4', linewidth=2.5, marker='o', markersize=8)
    ax.fill_between(range(len(acc_vals)), acc_vals, alpha=0.12, color='#1f77b4')

    ax.axhline(80, color='#ff7f0e', linewidth=1.5, linestyle='--', label='80% threshold')
    ax.axhline(50, color='#d62728', linewidth=1.5, linestyle='--', label='50% threshold')

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yticks(range(0, 101, 10))
    ax.set_yticklabels([f'{v}%' for v in range(0, 101, 10)], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Noise level (SNR)', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Robustness to White Noise', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', color='#e0e0e0', linewidth=0.8)
    for sp in ax.spines.values():
        sp.set_color('#cccccc')

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'noise_robustness.png')
    plt.savefig(plot_path, dpi=150, facecolor='white')
    print(f'Plot saved: {plot_path}')
    plt.show()


if __name__ == '__main__':
    main()
