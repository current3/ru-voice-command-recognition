import os
import sys
import time
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

sys.path.insert(0, os.path.dirname(__file__))
from config import SAMPLE_RATE, WHISPER_MODEL

ONNX_DIR = os.path.join(os.path.dirname(__file__), 'models', 'whisper_onnx')
ONNX_FILE = os.path.join(ONNX_DIR, 'model.onnx')
OPSET = 14


class _EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features).last_hidden_state


def export():
    print(f'Loading {WHISPER_MODEL}...')
    kw = dict(local_files_only=True)
    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL, **kw)
        model = WhisperModel.from_pretrained(
            WHISPER_MODEL, attn_implementation='eager', **kw
        )
    except OSError:
        print('  Cache not found, downloading...')
        feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
        model = WhisperModel.from_pretrained(WHISPER_MODEL, attn_implementation='eager')

    model.eval()
    wrapper = _EncoderWrapper(model.encoder)

    dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    inputs = feature_extractor(dummy_audio, sampling_rate=SAMPLE_RATE, return_tensors='pt')
    input_features = inputs['input_features']
    print(f'Input shape: {tuple(input_features.shape)}')

    os.makedirs(ONNX_DIR, exist_ok=True)
    feature_extractor.save_pretrained(ONNX_DIR)

    print(f'Exporting to ONNX (opset {OPSET})...')
    t0 = time.monotonic()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_features,),
            ONNX_FILE,
            input_names=['input_features'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_features':    {0: 'batch'},
                'last_hidden_state': {0: 'batch'},
            },
            opset_version=OPSET,
            do_constant_folding=True,
        )
    elapsed = time.monotonic() - t0
    size_mb = os.path.getsize(ONNX_FILE) / 1024 ** 2
    print(f'Done in {elapsed:.1f}s  |  {ONNX_FILE}  ({size_mb:.0f} MB)')

    print('Verifying with onnxruntime...')
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_FILE, providers=['CPUExecutionProvider'])
    out = sess.run(['last_hidden_state'], {'input_features': input_features.numpy()})
    print(f'OK — output shape: {out[0].shape}')
    print('\nNext steps:  python train.py  →  python realtime.py')


if __name__ == '__main__':
    export()
