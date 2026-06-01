import os
import sys
import re
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATASET_PATH, MODEL_PATH, LABELS_PATH,
    EMBEDDING_DIM, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    TEST_SPEAKERS, SAMPLE_RATE, WHISPER_MODEL,
)
from core.preprocess import load_audio, augment_audio, extract_embedding

_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    for _gpu in _gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
    print(f'[TF] GPU: {[g.name for g in _gpus]}')
else:
    print('[TF] CPU mode')

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'models', 'embedding_cache')
CACHE_VERSION = 'v4'
AUG_COUNT = 6


def get_speaker(filepath: str) -> str:
    stem = re.sub(r'_orig$', '', Path(filepath).stem)
    if stem.startswith('tts_'):
        return 'tts'
    return stem.lower()


def collect_files_by_speaker(dataset_path: str):
    train_files, train_labels = [], []
    test_files,  test_labels  = [], []
    tts_files,   tts_labels   = [], []

    for class_dir in sorted(Path(dataset_path).iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.glob('*'):
            if f.suffix.lower() not in ('.ogg', '.wav', '.mp3', '.flac'):
                continue
            spk = get_speaker(str(f))
            if spk == 'tts':
                tts_files.append(str(f)); tts_labels.append(class_dir.name)
            elif spk in TEST_SPEAKERS:
                test_files.append(str(f)); test_labels.append(class_dir.name)
            else:
                train_files.append(str(f)); train_labels.append(class_dir.name)

    return train_files, train_labels, test_files, test_labels, tts_files, tts_labels


def _cache_path(audio_path: str, suffix: str = '') -> str:
    p = Path(audio_path)
    return os.path.join(CACHE_DIR, p.parent.name, p.stem + suffix + '.npy')


def _cache_meta_path(cache_path: str) -> str:
    return cache_path + '.json'


def _cache_metadata(audio_path: str, cache_path: str) -> dict:
    source = Path(audio_path).resolve()
    stat = source.stat()
    return {
        'cache_version': CACHE_VERSION,
        'cache_key': Path(cache_path).name,
        'source_path': str(source),
        'source_size': stat.st_size,
        'source_mtime_ns': stat.st_mtime_ns,
        'sample_rate': SAMPLE_RATE,
        'wav2vec2_model': WHISPER_MODEL,
        'embedding_dim': EMBEDDING_DIM,
    }


def _load_cached(audio_path: str, cache_path: str) -> np.ndarray | None:
    if not os.path.exists(cache_path) or not os.path.exists(_cache_meta_path(cache_path)):
        return None
    try:
        with open(_cache_meta_path(cache_path), 'r', encoding='utf-8') as f:
            saved = json.load(f)
        if saved != _cache_metadata(audio_path, cache_path):
            return None
        emb = np.load(cache_path)
        return emb.astype(np.float32) if emb.shape == (EMBEDDING_DIM,) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _save_cached(audio_path: str, cache_path: str, emb: np.ndarray):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, emb)
    with open(_cache_meta_path(cache_path), 'w', encoding='utf-8') as f:
        json.dump(_cache_metadata(audio_path, cache_path), f, ensure_ascii=False, indent=2)


def build_features(files: list, labels: list, augment: bool = False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    X, y = [], []
    total = len(files)
    hits = 0

    for i, (path, label) in enumerate(zip(files, labels)):
        print(f'\r  [{i+1}/{total}] {label:<20}', end='', flush=True)

        cp = _cache_path(path)
        emb = _load_cached(path, cp)
        if emb is not None:
            hits += 1
            audio = None
        else:
            audio = load_audio(path)
            emb = extract_embedding(audio)
            _save_cached(path, cp, emb)

        X.append(emb); y.append(label)

        if augment:
            aug_cps = [_cache_path(path, f'_aug{j}') for j in range(AUG_COUNT)]
            aug_embs = [_load_cached(path, cp) for cp in aug_cps]
            if any(e is None for e in aug_embs):
                if audio is None:
                    audio = load_audio(path)
                for j, (aug, aug_emb) in enumerate(zip(augment_audio(audio), aug_embs)):
                    if aug_emb is None:
                        aug_emb = extract_embedding(aug)
                        _save_cached(path, aug_cps[j], aug_emb)
                        aug_embs[j] = aug_emb
            for aug_emb in aug_embs:
                X.append(aug_emb); y.append(label)

    print(f'  (cache: {hits}/{total})')
    return np.array(X, dtype=np.float32), np.array(y)


def build_model(num_classes: int) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(1e-4)
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(EMBEDDING_DIM,)),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ], name='voice_classifier')


def main():
    print(f'Dataset: {DATASET_PATH}')

    train_files, train_labels, test_files, test_labels, tts_files, tts_labels = \
        collect_files_by_speaker(DATASET_PATH)

    class_names = sorted(set(train_labels + test_labels))
    num_classes = len(class_names)
    label2idx = {name: i for i, name in enumerate(class_names)}

    train_speakers = set(get_speaker(f) for f in train_files)
    print(f'Classes: {num_classes}')
    print(f'Train speakers ({len(train_speakers)}): {len(train_files)} files')
    print(f'Test speakers {TEST_SPEAKERS}: {len(test_files)} files')

    split = train_test_split(
        train_files, train_labels, test_size=0.20,
        stratify=train_labels, random_state=42,
    )
    train_files, val_files, train_labels, val_labels = split
    print(f'Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}')

    print('\nExtracting train features (with augmentation):')
    X_train, y_train = build_features(train_files, train_labels, augment=True)
    print('Extracting val features:')
    X_val, y_val = build_features(val_files, val_labels, augment=False)
    print('Extracting test features:')
    X_test, y_test = build_features(test_files, test_labels, augment=False)

    y_train_idx = np.array([label2idx[l] for l in y_train])
    y_val_idx   = np.array([label2idx[l] for l in y_val])
    y_test_idx  = np.array([label2idx[l] for l in y_test])

    y_train_oh = tf.keras.utils.to_categorical(y_train_idx, num_classes)
    y_val_oh   = tf.keras.utils.to_categorical(y_val_idx,   num_classes)
    y_test_oh  = tf.keras.utils.to_categorical(y_test_idx,  num_classes)

    print(f'\nTrain shape: {X_train.shape}')

    model = build_model(num_classes)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy'],
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    np.save(LABELS_PATH, np.array(class_names))

    model.load_weights(MODEL_PATH)
    _, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f'\nTest accuracy: {test_acc*100:.2f}%')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('white')
    axes[0].plot(history.history['loss'],     color='#1f77b4', linewidth=2, label='Train')
    axes[0].plot(history.history['val_loss'], color='#ff7f0e', linewidth=2, label='Val')
    axes[0].set_title('Loss', fontsize=12)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, color='#e0e0e0')
    axes[1].plot(history.history['accuracy'],     color='#1f77b4', linewidth=2, label='Train')
    axes[1].plot(history.history['val_accuracy'], color='#ff7f0e', linewidth=2, label='Val')
    axes[1].set_title('Accuracy', fontsize=12)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, color='#e0e0e0')
    for ax in axes:
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_color('#cccccc')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(MODEL_PATH), 'training_history.png'),
                dpi=150, facecolor='white')
    plt.show()

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test_idx, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('white')
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix', pad=16, fontsize=13)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(os.path.dirname(MODEL_PATH), 'confusion_matrix.png'),
                dpi=150, facecolor='white')
    plt.show()

    if tts_files:
        print('\nExtracting TTS features:')
        X_tts, y_tts = build_features(tts_files, tts_labels, augment=False)
        y_tts_idx = np.array([label2idx[l] for l in y_tts])
        y_tts_oh  = tf.keras.utils.to_categorical(y_tts_idx, num_classes)
        _, tts_acc = model.evaluate(X_tts, y_tts_oh, verbose=0)
        print(f'TTS accuracy: {tts_acc*100:.2f}%')


if __name__ == '__main__':
    main()
