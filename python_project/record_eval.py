import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import SAMPLE_RATE

EVAL_SPEAKER_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset_eval')

COMMANDS = [
    'вперед',     'назад',       'влево',
    'вправо',     'стоп',        'один',
    'два',        'три',         'режим',
    'включить',   'выключить',   'поверни',
    'налево',     'направо',     'двигайся',
    'развернись', 'остановись',  'автоматический',
    'ручной',     'домой',       'быстрее',
    'медленнее',  'разворот',    'вернись',
]

RECORD_SECONDS = 1.5


def count_existing(command: str) -> int:
    folder = Path(EVAL_SPEAKER_DIR) / command
    if not folder.exists():
        return 0
    return len([f for f in folder.iterdir()
                if f.suffix.lower() in ('.wav', '.ogg', '.mp3', '.flac')])


def next_filename(command: str) -> Path:
    folder = Path(EVAL_SPEAKER_DIR) / command
    folder.mkdir(parents=True, exist_ok=True)
    idx = 1
    while (folder / f'spk_{idx:03d}.wav').exists():
        idx += 1
    return folder / f'spk_{idx:03d}.wav'


def record_audio() -> np.ndarray:
    frames = int(SAMPLE_RATE * RECORD_SECONDS)
    for i in (3, 2, 1):
        print(f'\r{i}...', end='', flush=True)
        time.sleep(0.6)
    print('\rSpeak now', flush=True)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1,
                   dtype='float32', blocking=True)
    return audio.flatten()


def play_audio(audio: np.ndarray):
    sd.play(audio, samplerate=SAMPLE_RATE, blocking=True)


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return 'q'


def main():
    Path(EVAL_SPEAKER_DIR).mkdir(parents=True, exist_ok=True)
    print(f'Eval speaker recorder  |  {Path(EVAL_SPEAKER_DIR).resolve()}')
    print('Speaker must NOT appear in training data.\n')
    ask('Press ENTER to start...')

    saved = 0
    i = 0
    while i < len(COMMANDS):
        cmd = COMMANDS[i]
        n = count_existing(cmd)
        print(f'\n[{i+1}/{len(COMMANDS)}] {cmd}  ({n} recordings)')
        action = ask('ENTER record  r retry  s skip  q quit\n> ')

        if action == 'q':
            break
        if action == 's':
            i += 1
            continue

        try:
            audio = record_audio()
        except Exception as e:
            print(f'Error: {e}')
            ask('Press ENTER...')
            continue

        play_audio(audio)
        decision = ask('Save? ENTER yes  r retry  s skip\n> ')

        if decision == 'r':
            continue
        if decision == 's':
            i += 1
            continue

        path = next_filename(cmd)
        sf.write(str(path), audio, SAMPLE_RATE, subtype='PCM_16')
        saved += 1
        print(f'Saved: {path.name}')
        time.sleep(0.4)
        i += 1

    print(f'\nDone. Saved {saved} files this session.')
    print('Next: python evaluate.py')


if __name__ == '__main__':
    main()
