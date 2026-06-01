import os
import numpy as np
import librosa
import torch
from transformers import WhisperFeatureExtractor, WhisperModel, logging as hf_logging

hf_logging.set_verbosity_error()

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

from config import SAMPLE_RATE, WHISPER_MODEL

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_ONNX_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'models', 'whisper_onnx'))

N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160

_feature_extractor = None
_model = None
_ort_model = None


def get_model():
    global _feature_extractor, _model, _ort_model
    if _feature_extractor is not None:
        return _feature_extractor, _model

    print(f'Loading {WHISPER_MODEL}...')
    onnx_file = os.path.join(_ONNX_DIR, 'model.onnx')
    if os.path.exists(onnx_file):
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            _ort_model = ort.InferenceSession(
                onnx_file, sess_options=opts, providers=['CPUExecutionProvider']
            )
            _feature_extractor = WhisperFeatureExtractor.from_pretrained(_ONNX_DIR)
            print('Whisper encoder loaded via ONNX Runtime.')
            return _feature_extractor, None
        except Exception as e:
            print(f'ONNX failed ({e}), falling back to PyTorch.')

    kw = dict(local_files_only=True)
    try:
        _feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL, **kw)
        _model = WhisperModel.from_pretrained(WHISPER_MODEL, **kw)
    except OSError:
        print('  Cache not found, downloading...')
        _feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
        _model = WhisperModel.from_pretrained(WHISPER_MODEL)

    _model.eval()
    if _device.type == 'cpu':
        _model = torch.quantization.quantize_dynamic(
            _model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print('Whisper loaded via PyTorch (CPU INT8).')
    else:
        _model = _model.to(_device)
        print(f'Whisper loaded via PyTorch (GPU: {torch.cuda.get_device_name(0)}).')

    return _feature_extractor, _model


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32)


def pad_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    min_samples = int(1.0 * sr)
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))
    return audio


def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    target = len(audio)
    if len(stretched) > target:
        return stretched[:target].astype(np.float32)
    return np.pad(stretched, (0, target - len(stretched))).astype(np.float32)


def augment_audio(audio: np.ndarray) -> list[np.ndarray]:
    results = []
    results.append((audio * 0.4).astype(np.float32))
    shift = int(0.03 * len(audio))
    results.append(np.concatenate([np.zeros(shift, dtype=np.float32), audio[:-shift]]))
    for sigma in (0.005, 0.015):
        noisy = audio + np.random.randn(len(audio)).astype(np.float32) * sigma
        results.append(np.clip(noisy, -1.0, 1.0))
    for rate in (0.85, 1.15):
        results.append(_time_stretch(audio, rate))
    return results


def _pool_speech_frames(hidden: np.ndarray, audio_len: int) -> np.ndarray:
    total_frames = hidden.shape[1]
    speech_frames = max(1, min(int(audio_len / SAMPLE_RATE * 50), total_frames))
    return hidden[:, :speech_frames, :].mean(axis=1).squeeze()


def extract_embedding(audio: np.ndarray) -> np.ndarray:
    feature_extractor, model = get_model()
    audio = pad_audio(audio)
    inputs = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors='pt')

    if _ort_model is not None:
        out = _ort_model.run(
            ['last_hidden_state'],
            {'input_features': inputs['input_features'].numpy()},
        )
        return _pool_speech_frames(out[0], len(audio)).astype(np.float32)

    if _device.type != 'cpu':
        inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        hidden = model.encoder(inputs['input_features']).last_hidden_state.cpu().numpy()
    return _pool_speech_frames(hidden, len(audio)).astype(np.float32)


def extract_log_mel(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    return librosa.power_to_db(mel, ref=np.max)
