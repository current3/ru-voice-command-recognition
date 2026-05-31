import os
import sys
import numpy as np
import soundfile as sf
import torch
import librosa
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import DATASET_PATH, SAMPLE_RATE

TTS_RATE = 24000

try:
    from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain
except ImportError:
    print('Install audiomentations: pip install audiomentations')
    sys.exit(1)

COMMANDS = [
    'вперед',        'назад',         'влево',
    'вправо',        'стоп',          'один',
    'два',           'три',           'режим',
    'включить',      'выключить',     'поверни',
    'налево',        'направо',       'двигайся',
    'развернись',    'остановись',    'автоматический',
    'ручной',        'домой',         'быстрее',
    'медленнее',     'разворот',      'вернись',
]

SPEAKERS = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene']
NUM_AUG = 5


def load_tts():
    import tempfile
    print('Loading Silero TTS v4_ru...')
    orig_dir = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v4_ru',
            trust_repo=True,
        )
    finally:
        os.chdir(orig_dir)
    return model


def tts_audio(model, text: str, speaker: str) -> np.ndarray:
    with torch.no_grad():
        tensor = model.apply_tts(
            text=text, speaker=speaker,
            sample_rate=TTS_RATE,
            put_accent=True, put_yo=True,
        )
    audio = tensor.numpy().astype(np.float32)
    return librosa.resample(audio, orig_sr=TTS_RATE, target_sr=SAMPLE_RATE)


def make_augmentor() -> Compose:
    return Compose([
        Gain(min_gain_db=-8, max_gain_db=8, p=0.7),
        AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.025, p=0.6),
        TimeStretch(min_rate=0.80, max_rate=1.20, p=0.5),
        PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
    ])


def save(path: str, audio: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, SAMPLE_RATE, subtype='PCM_16')


def main():
    dataset_root = Path(DATASET_PATH)
    model = load_tts()
    augment = make_augmentor()

    files_per_class = len(SPEAKERS) * (1 + NUM_AUG)
    total = len(COMMANDS) * files_per_class
    print(f'{len(COMMANDS)} commands × {files_per_class} files = {total} total\n')

    generated = 0
    for cmd in COMMANDS:
        class_dir = dataset_root / cmd
        class_dir.mkdir(parents=True, exist_ok=True)

        for spk in SPEAKERS:
            audio = tts_audio(model, cmd, spk)
            save(str(class_dir / f'tts_{spk}_orig.wav'), audio)
            generated += 1
            for i in range(NUM_AUG):
                aug = augment(samples=audio, sample_rate=SAMPLE_RATE)
                save(str(class_dir / f'tts_{spk}_aug{i}.wav'), aug)
                generated += 1

        count = len(list(class_dir.glob('*.wav')))
        print(f'[{generated}/{total}] {cmd:<20} {count} files')

    print(f'\nDone. Next: add real recordings to dataset/<command>/, then python train.py')


if __name__ == '__main__':
    main()
