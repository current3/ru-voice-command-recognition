import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(__file__))
from config import MODEL_PATH, LABELS_PATH
from core.preprocess import load_audio, extract_embedding

EVAL_SPEAKER_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset_eval')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def collect_files(root: str):
    files, labels = [], []
    for class_dir in sorted(Path(root).iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.glob('*'):
            if f.suffix.lower() in ('.wav', '.ogg', '.mp3', '.flac'):
                files.append(str(f))
                labels.append(class_dir.name)
    return files, labels


def main():
    if not Path(EVAL_SPEAKER_DIR).exists():
        print('dataset_eval/ not found.')
        print('Run first: python record_eval.py')
        return

    print('Loading model...')
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(LABELS_PATH, allow_pickle=True).tolist()

    files, labels = collect_files(EVAL_SPEAKER_DIR)
    if not files:
        print('No files found in dataset_eval/.')
        return

    print(f'New speaker files: {len(files)}\n')

    label2idx = {n: i for i, n in enumerate(class_names)}
    y_true, y_pred = [], []
    per_class = {c: [0, 0] for c in class_names}

    for i, (path, label) in enumerate(zip(files, labels)):
        print(f'\r  [{i+1}/{len(files)}] {label:<22}', end='', flush=True)
        audio = load_audio(path)
        emb = extract_embedding(audio)
        probs = model(emb[np.newaxis], training=False).numpy()[0]
        pred = class_names[int(probs.argmax())]
        true_idx = label2idx.get(label)
        if true_idx is None:
            continue
        y_true.append(true_idx)
        y_pred.append(int(probs.argmax()))
        per_class[label][1] += 1
        if pred == label:
            per_class[label][0] += 1

    print()

    if not y_true:
        print('No files with known labels found.')
        print('Check folder names in dataset_eval/ against labels.npy.')
        return

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    print(f'\n{"Command":<22} {"Correct":>7} {"Total":>6} {"Accuracy":>10}')
    print('-' * 50)
    for cmd in sorted(per_class.keys()):
        correct, total = per_class[cmd]
        if total == 0:
            continue
        acc = correct / total * 100
        mark = '' if acc == 100 else '  <--'
        print(f'  {cmd:<20} {correct:>7} {total:>6} {acc:>9.1f}%{mark}')

    print('-' * 50)
    print(f'  {"TOTAL":<20} '
          f'{sum(v[0] for v in per_class.values()):>7} '
          f'{len(y_true):>6} '
          f'{accuracy*100:>9.1f}%')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'new_speaker_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['command', 'correct', 'total', 'accuracy'])
        for cmd in sorted(per_class.keys()):
            correct, total = per_class[cmd]
            if total > 0:
                w.writerow([cmd, correct, total, f'{correct/total:.4f}'])
    print(f'\nCSV saved: {csv_path}')

    tested_classes = sorted({labels[i] for i in range(len(labels))
                              if label2idx.get(labels[i]) is not None})
    tested_indices = [label2idx[c] for c in tested_classes]
    cm = confusion_matrix(y_true, y_pred, labels=tested_indices)

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('white')
    disp = ConfusionMatrixDisplay(cm, display_labels=tested_classes)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — new speaker (accuracy {accuracy*100:.1f}%)',
                 fontsize=13, pad=16)
    ax.tick_params(axis='both', labelsize=9)
    plt.tight_layout(pad=2.0)
    plot_path = os.path.join(RESULTS_DIR, 'new_speaker_confusion.png')
    plt.savefig(plot_path, dpi=150, facecolor='white')
    print(f'Plot saved: {plot_path}')
    plt.show()


if __name__ == '__main__':
    main()
