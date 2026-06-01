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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def get_speaker(filepath: str) -> str:
    stem = re.sub(r'_orig$', '', Path(filepath).stem)
    return 'tts' if stem.startswith('tts_') else stem.lower()


def collect_by_group(dataset_path: str):
    train_files, train_labels = [], []
    test_files,  test_labels  = [], []
    tts_files,   tts_labels   = [], []

    for class_dir in sorted(Path(dataset_path).iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.glob('*'):
            if f.suffix.lower() not in ('.wav', '.ogg', '.mp3', '.flac'):
                continue
            spk = get_speaker(str(f))
            if spk == 'tts':
                tts_files.append(str(f)); tts_labels.append(class_dir.name)
            elif spk in TEST_SPEAKERS:
                test_files.append(str(f)); test_labels.append(class_dir.name)
            else:
                train_files.append(str(f)); train_labels.append(class_dir.name)

    return train_files, train_labels, test_files, test_labels, tts_files, tts_labels


def evaluate(model, class_names, files, labels):
    if not files:
        return 0.0, {}
    per_class = {c: [0, 0] for c in class_names}
    total = len(files)
    for i, (path, label) in enumerate(zip(files, labels)):
        print(f'\r  [{i+1}/{total}]', end='', flush=True)
        emb = extract_embedding(load_audio(path))
        probs = model(emb[np.newaxis], training=False).numpy()[0]
        pred = class_names[int(probs.argmax())]
        per_class[label][1] += 1
        if pred == label:
            per_class[label][0] += 1
    print()
    overall = sum(v[0] for v in per_class.values()) / len(files)
    per_acc = {c: (v[0]/v[1] if v[1] else 0.0)
               for c, v in per_class.items() if v[1] > 0}
    return overall, per_acc


def plot_comparison(groups: dict):
    names = list(groups.keys())
    accs = [groups[n][0] * 100 for n in names]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')

    ax = axes[0]
    ax.set_facecolor('white')
    bars = ax.bar(names, accs, color=colors[:len(names)], width=0.4)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_ylabel('Overall accuracy (%)', fontsize=11)
    ax.set_title('Accuracy by speaker group', fontsize=12)
    ax.grid(axis='y', color='#e0e0e0')
    for sp in ax.spines.values():
        sp.set_color('#cccccc')

    ax2 = axes[1]
    ax2.set_facecolor('white')
    all_classes = sorted(set(c for _, per in groups.values() for c in per))
    x = np.arange(len(all_classes))
    w = 0.8 / len(names)
    for i, (name, (_, per)) in enumerate(groups.items()):
        vals = [per.get(c, 0) * 100 for c in all_classes]
        ax2.bar(x + i*w - w*(len(names)-1)/2, vals, w,
                label=name, color=colors[i], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_classes, rotation=90, fontsize=8)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Per-class accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', color='#e0e0e0')
    for sp in ax2.spines.values():
        sp.set_color('#cccccc')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'speaker_transfer.png')
    plt.savefig(path, dpi=150, facecolor='white')
    print(f'Plot saved: {path}')
    plt.show()


def main():
    print('Loading model...')
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(LABELS_PATH, allow_pickle=True).tolist()

    train_files, train_labels, test_files, test_labels, tts_files, tts_labels = \
        collect_by_group(DATASET_PATH)

    print(f'Train speakers: {len(train_files)} files')
    print(f'Test speakers {TEST_SPEAKERS}: {len(test_files)} files')
    print(f'TTS: {len(tts_files)} files\n')

    groups = {}

    print('Evaluating train speakers...')
    acc_train, per_train = evaluate(model, class_names, train_files, train_labels)
    print(f'  Accuracy: {acc_train*100:.1f}%')
    groups['Train\nspeakers'] = (acc_train, per_train)

    print(f'Evaluating test speakers ({TEST_SPEAKERS})...')
    acc_test, per_test = evaluate(model, class_names, test_files, test_labels)
    print(f'  Accuracy: {acc_test*100:.1f}%')
    groups[f'Unseen speakers\n({", ".join(TEST_SPEAKERS)})'] = (acc_test, per_test)

    per_tts = {}
    if tts_files:
        print('Evaluating TTS...')
        acc_tts, per_tts = evaluate(model, class_names, tts_files, tts_labels)
        print(f'  Accuracy: {acc_tts*100:.1f}%')
        groups['Synthetic\n(TTS)'] = (acc_tts, per_tts)

    drop = (acc_train - acc_test) * 100
    print(f'\nAccuracy drop (train → unseen): {drop:+.1f}%')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'speaker_transfer.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['class', 'acc_train', 'acc_test', 'acc_tts'])
        for c in sorted(set(per_train) | set(per_test)):
            w.writerow([c,
                        f'{per_train.get(c, 0):.4f}',
                        f'{per_test.get(c, 0):.4f}',
                        f'{per_tts.get(c, 0):.4f}' if tts_files else ''])
    print(f'CSV saved: {csv_path}')

    plot_comparison(groups)


if __name__ == '__main__':
    main()
