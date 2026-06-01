import os
import sys
import csv
import queue
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.gridspec as gridspec
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    MODEL_PATH, LABELS_PATH, SAMPLE_RATE,
    SILENCE_THRESHOLD, CONFIDENCE_THRESHOLD, CONFIDENCE_OVERRIDES, UNKNOWN_CLASS,
)

CHUNK_SAMPLES = SAMPLE_RATE // 10
DISPLAY_SAMPLES = int(SAMPLE_RATE * 1.5)
WAVE_POINTS = 3000
MAX_SEGMENT_SAMPLES = int(SAMPLE_RATE * 2.5)
NUM_TOP = 5
MAX_HISTORY = 8
DEBOUNCE_SEC = 0.8
END_SILENCE_SEC = 0.40
MIN_SPEECH_SEC = 0.20
PRE_ROLL_SEC = 0.15
VAD_CALIBRATION_SEC = 1.5
VAD_NOISE_MULT = 3.0
VU_MAX_RMS = 0.3
RESULT_HOLD_SEC = 2.5
GUI_FPS = 12
DEBUG_SEGMENTS = False

_GUI_INTERVAL = 1.0 / GUI_FPS
_MEL_FRAMES = 1 + (DISPLAY_SAMPLES - 400) // 160

fig = plt.figure(figsize=(16, 9), facecolor='white')
fig.canvas.manager.set_window_title('Voice-Controlled Robot')

_ax_splash = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor='white')
_ax_splash.axis('off')
_ax_splash.text(0.5, 0.62, 'Voice-Controlled Robot',
                ha='center', va='center', fontsize=24, fontweight='bold',
                color='#1e293b', transform=_ax_splash.transAxes)
_txt_status_load = _ax_splash.text(
    0.5, 0.50, '⟳ Loading classifier...',
    ha='center', va='center', fontsize=14,
    color='#2563eb', transform=_ax_splash.transAxes)
_ax_splash.text(0.5, 0.40, 'This may take a few seconds',
                ha='center', va='center', fontsize=10,
                color='#94a3b8', transform=_ax_splash.transAxes)
plt.pause(0.05)

classifier = None
class_names = None
_load_done = False
_load_error = ''


def _load_models():
    global classifier, class_names, _load_done, _load_error
    try:
        import tensorflow as tf
        classifier = tf.keras.models.load_model(MODEL_PATH)
        class_names = np.load(LABELS_PATH, allow_pickle=True).tolist()
        from core.preprocess import get_model
        get_model()
    except Exception as e:
        _load_error = str(e)
    finally:
        _load_done = True


_load_thread = threading.Thread(target=_load_models, daemon=True)
_load_thread.start()

_dots = 0
while not _load_done:
    _dots = (_dots + 1) % 4
    try:
        base = _txt_status_load.get_text().rstrip('. ')
        _txt_status_load.set_text(base + '.' * _dots)
    except Exception:
        pass
    plt.pause(0.3)

if _load_error:
    _txt_status_load.set_text(f'Load error:\n{_load_error}')
    _txt_status_load.set_color('#ff5555')
    plt.pause(10)
    sys.exit(1)

_ax_splash.remove()
del _ax_splash, _txt_status_load

from core.preprocess import extract_embedding, extract_log_mel
from core.robot_viz import RobotVisualizer
from core.intent_engine import IntentEngine
from core.grid_map import GridMap
from core.autopilot import AutoPilot

print(f'Ready. Classes: {len(class_names)}. Speak a command...\n')


class VAD:
    CALIBRATING = 'calibrating'
    LISTENING = 'listening'
    RECORDING = 'recording'

    def __init__(self):
        self.state = self.CALIBRATING
        self.threshold = SILENCE_THRESHOLD
        self._calib_n = max(1, int(VAD_CALIBRATION_SEC * SAMPLE_RATE / CHUNK_SAMPLES))
        self._end_samp = int(END_SILENCE_SEC * SAMPLE_RATE)
        self._min_samp = int(MIN_SPEECH_SEC * SAMPLE_RATE)
        self._pre_samp = int(PRE_ROLL_SEC * SAMPLE_RATE)
        self._noise_buf: list[float] = []
        self._pre_roll: list[np.ndarray] = []
        self._speech: list[np.ndarray] = []
        self._sil_acc = 0

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
        if self.state == self.CALIBRATING:
            return self._calibrate(chunk, rms)
        if self.state == self.LISTENING:
            return self._listen(chunk, rms)
        return self._record(chunk, rms)

    def is_calibrating(self):
        return self.state == self.CALIBRATING

    def is_recording(self):
        return self.state == self.RECORDING

    def _calibrate(self, _chunk, rms):
        self._noise_buf.append(rms)
        if len(self._noise_buf) >= self._calib_n:
            nf = float(np.percentile(self._noise_buf, 90))
            self.threshold = max(SILENCE_THRESHOLD * 2.5, nf * VAD_NOISE_MULT)
            print(f'[VAD] threshold={self.threshold:.5f}  noise={nf:.5f}')
            self.state = self.LISTENING
        return None

    def _listen(self, chunk, rms):
        if rms >= self.threshold:
            self._speech = [c.copy() for c in self._pre_roll] + [chunk.copy()]
            self._sil_acc = 0
            self._pre_roll = []
            self.state = self.RECORDING
            print(f'[VAD] speech  rms={rms:.4f}')
        else:
            self._pre_roll.append(chunk.copy())
            while sum(len(c) for c in self._pre_roll) > self._pre_samp:
                self._pre_roll.pop(0)
        return None

    def _record(self, chunk, rms):
        self._speech.append(chunk.copy())
        self._sil_acc = 0 if rms >= self.threshold else self._sil_acc + len(chunk)
        total = sum(len(c) for c in self._speech)
        if self._sil_acc >= self._end_samp or total >= MAX_SEGMENT_SAMPLES:
            seg = np.concatenate(self._speech).astype(np.float32)
            self._speech = []; self._sil_acc = 0; self._pre_roll = []
            self.state = self.LISTENING
            print(f'[VAD] segment {len(seg)/SAMPLE_RATE:.2f}s')
            return seg if len(seg) >= self._min_samp else None
        return None


class InferenceWorker:
    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._t0: float | None = None
        self.last_probs: np.ndarray | None = None
        self.last_latency_ms: float = 0.0

    def submit(self, audio: np.ndarray) -> bool:
        if self._future is not None:
            return False
        self._t0 = time.monotonic()
        self._future = self._pool.submit(self._run, audio.copy())
        return True

    def busy(self):
        return self._future is not None

    def poll(self) -> np.ndarray | None:
        if self._future is None or not self._future.done():
            return None
        try:
            probs = self._future.result()
            self.last_latency_ms = (time.monotonic() - self._t0) * 1000
            self.last_probs = probs
            best = class_names[int(probs.argmax())]
            print(f'[INF] {best} ({probs.max()*100:.1f}%)  {self.last_latency_ms:.0f}ms')
            return probs
        except Exception as e:
            print(f'[INF] error: {e}')
            return None
        finally:
            self._future = None
            self._t0 = None

    @staticmethod
    def _run(audio: np.ndarray) -> np.ndarray:
        emb = extract_embedding(audio)
        return classifier(emb[np.newaxis], training=False).numpy()[0]


class SessionLogger:
    LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

    def __init__(self):
        self._rows: list[dict] = []
        self._t0 = datetime.now()

    def log(self, cmd, conf, lat, triggered):
        self._rows.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'command': cmd,
            'confidence': round(float(conf), 4),
            'latency_ms': round(lat, 1),
            'triggered': triggered,
        })

    def save(self):
        if not self._rows:
            return
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        p = os.path.join(self.LOGS_DIR, f"session_{self._t0.strftime('%Y%m%d_%H%M%S')}.csv")
        with open(p, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=self._rows[0].keys())
            w.writeheader(); w.writerows(self._rows)
        print(f'Log: {p}')
        self._save_confidence_chart(p)

    def _save_confidence_chart(self, csv_path: str):
        triggered = [r for r in self._rows if r['triggered']]
        if not triggered:
            return

        conf_sum = defaultdict(float)
        conf_count = defaultdict(int)
        for r in triggered:
            conf_sum[r['command']] += r['confidence']
            conf_count[r['command']] += 1

        classes = sorted(conf_count.keys())
        avg_conf = [conf_sum[c] / conf_count[c] for c in classes]
        counts = [conf_count[c] for c in classes]
        total = len(triggered)
        avg_lat = sum(r['latency_ms'] for r in self._rows) / len(self._rows)

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(classes) * 0.4 + 2)))
        fig.patch.set_facecolor('white')
        fig.suptitle(
            f'Session stats  |  commands: {total}  |  avg latency: {avg_lat:.0f} ms',
            fontsize=12, fontweight='bold', color='#1e293b', y=1.01,
        )

        ax1 = axes[0]
        ax1.set_facecolor('white')
        ax1.set_title('Average confidence per class', fontsize=10,
                      fontweight='bold', color='#475569')
        colors = ['#16a34a' if v >= 0.65 else '#d97706' if v >= 0.45 else '#dc2626'
                  for v in avg_conf]
        bars = ax1.barh(classes, avg_conf, color=colors, height=0.6)
        ax1.set_xlim(0, 1)
        ax1.axvline(0.65, color='#16a34a', linewidth=1, linestyle='--', alpha=0.5)
        ax1.axvline(0.45, color='#d97706', linewidth=1, linestyle='--', alpha=0.5)
        for bar, val in zip(bars, avg_conf):
            ax1.text(min(val + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                     f'{val*100:.0f}%', va='center', fontsize=8, color='#1e293b')
        ax1.set_xlabel('Confidence', fontsize=9, color='#475569')
        ax1.set_ylabel('Command class', fontsize=9, color='#475569')
        ax1.tick_params(axis='y', labelsize=9, colors='#334155')
        ax1.tick_params(axis='x', labelsize=8, colors='#94a3b8')
        for sp in ax1.spines.values():
            sp.set_color('#e2e8f0')
        ax1.invert_yaxis()

        ax2 = axes[1]
        ax2.set_facecolor('white')
        ax2.set_title('Trigger count per class', fontsize=10,
                      fontweight='bold', color='#475569')
        ax2.barh(classes, counts, color='#3b82f6', height=0.6)
        for i, (_, n) in enumerate(zip(classes, counts)):
            ax2.text(n + 0.1, i, str(n), va='center', fontsize=8, color='#1e293b')
        ax2.set_xlabel('Count', fontsize=9, color='#475569')
        ax2.set_ylabel('Command class', fontsize=9, color='#475569')
        ax2.tick_params(axis='y', labelsize=9, colors='#334155')
        ax2.tick_params(axis='x', labelsize=8, colors='#94a3b8')
        for sp in ax2.spines.values():
            sp.set_color('#e2e8f0')
        ax2.invert_yaxis()

        plt.tight_layout()
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        out = csv_path.replace(self.LOGS_DIR, self.RESULTS_DIR).replace('.csv', '_confidence.png')
        plt.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        print(f'Confidence chart: {out}')


vad = VAD()
worker = InferenceWorker()
engine = IntentEngine()
logger = SessionLogger()
audio_q: queue.Queue = queue.Queue(maxsize=16)
_seg_q: queue.Queue = queue.Queue(maxsize=2)

_mel_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
_mel_busy: bool = False
_mel_cache: np.ndarray = np.full((64, _MEL_FRAMES), -50.0, dtype=np.float32)

_display_buf = np.zeros(DISPLAY_SAMPLES, dtype=np.float32)
_history: list[str] = []
_last_cmd = ''
_last_cmd_t = 0.0
_debug_idx = 0
_last_redraw = 0.0
_result_until = 0.0
_last_probs: np.ndarray | None = None

gs = gridspec.GridSpec(4, 3, figure=fig,
    height_ratios=[1.4, 1.0, 1.1, 0.28],
    left=0.05, right=0.97, top=0.95, bottom=0.05,
    hspace=0.55, wspace=0.38)

ax_robot = fig.add_subplot(gs[:, 0])
robot = RobotVisualizer(ax_robot)
grid_map = GridMap()
autopilot = AutoPilot(grid_map)
robot.setup_navigation(grid_map)

ax_cmd = fig.add_subplot(gs[0, 1])
ax_cmd.set_facecolor('#eff6ff'); ax_cmd.axis('off')
for sp in ax_cmd.spines.values():
    sp.set_edgecolor('#bfdbfe'); sp.set_visible(True)
txt_status = ax_cmd.text(0.5, 0.91, '● Calibrating', ha='center', va='top',
                          fontsize=9, color='#d97706', transform=ax_cmd.transAxes)
txt_cmd = ax_cmd.text(0.5, 0.54, 'Waiting...', ha='center', va='center',
                       fontsize=22, fontweight='bold', color='#64748b',
                       transform=ax_cmd.transAxes)
txt_conf = ax_cmd.text(0.5, 0.13, '', ha='center', va='center',
                        fontsize=12, color='#94a3b8', transform=ax_cmd.transAxes)

ax_hist = fig.add_subplot(gs[0, 2])
ax_hist.set_facecolor('#f8fafc'); ax_hist.axis('off')
ax_hist.set_title('Command History', color='#475569', fontsize=10, fontweight='bold')
txt_hist = ax_hist.text(0.07, 0.92, '', ha='left', va='top', fontsize=10,
                         color='#334155', transform=ax_hist.transAxes, linespacing=1.7)

ax_wave = fig.add_subplot(gs[1, 1])
ax_wave.set_facecolor('white')
ax_wave.set_title('Audio Signal', color='#475569', fontsize=9, fontweight='bold')
ax_wave.set_ylim(-1, 1)
ax_wave.tick_params(colors='#94a3b8', labelsize=7)
for sp in ax_wave.spines.values():
    sp.set_color('#e2e8f0')
wave_line, = ax_wave.plot(np.zeros(WAVE_POINTS), color='#2563eb', linewidth=0.7)

ax_mel = fig.add_subplot(gs[2, 1])
ax_mel.set_facecolor('white')
ax_mel.set_title('Mel Spectrogram', color='#475569', fontsize=9, fontweight='bold')
ax_mel.axis('off')
mel_img = ax_mel.imshow(np.full((64, _MEL_FRAMES), -50.0), aspect='auto',
                         origin='lower', cmap='Blues', vmin=-50, vmax=0)

ax_vu = fig.add_subplot(gs[3, 1])
ax_vu.set_facecolor('white'); ax_vu.set_xlim(0, 1); ax_vu.set_ylim(0, 1)
ax_vu.axis('off')
ax_vu.set_title('Volume', color='#475569', fontsize=8, fontweight='bold', pad=2)
ax_vu.barh(0.5, 1.0,  height=0.55, color='#f1f5f9', align='center', zorder=1)
ax_vu.barh(0.5, 0.60, height=0.55, left=0.0,  color='#dbeafe', align='center', zorder=2)
ax_vu.barh(0.5, 0.20, height=0.55, left=0.60, color='#fef9c3', align='center', zorder=2)
ax_vu.barh(0.5, 0.20, height=0.55, left=0.80, color='#fee2e2', align='center', zorder=2)
vu_bar = ax_vu.barh(0.5, 0.0, height=0.55, color='#2563eb', align='center', zorder=3)
_thr_x = SILENCE_THRESHOLD / VU_MAX_RMS
thr_line = ax_vu.axvline(_thr_x, color='#dc2626', linewidth=1.2, linestyle='--', zorder=4)
ax_vu.text(_thr_x + 0.01, 0.92, 'thr', color='#dc2626',
           fontsize=6, va='top', transform=ax_vu.transAxes)

ax_prob = fig.add_subplot(gs[1:, 2])
ax_prob.set_facecolor('white')
ax_prob.set_title('Top-5 Classes', color='#475569', fontsize=9, fontweight='bold')
ax_prob.set_xlim(0, 1)
ax_prob.tick_params(colors='#94a3b8', labelsize=9)
for sp in ax_prob.spines.values():
    sp.set_color('#e2e8f0')
prob_bars = ax_prob.barh(range(NUM_TOP), [0]*NUM_TOP, color=['#3b82f6']*NUM_TOP, height=0.55)
ax_prob.set_yticks(range(NUM_TOP))
ax_prob.set_yticklabels(['']*NUM_TOP, color='#334155')
ax_prob.invert_yaxis()


def _rms(a):
    return float(np.sqrt(np.mean(a**2))) if len(a) else 0.0


_AUTO_ALLOWED = {'стоп', 'остановись', 'ручной'}


def _should_trigger(cmd, conf):
    global _last_cmd, _last_cmd_t
    if robot.mode == 'auto' and cmd not in _AUTO_ALLOWED:
        return False
    threshold = CONFIDENCE_OVERRIDES.get(cmd, CONFIDENCE_THRESHOLD)
    if conf < threshold:
        return False
    now = time.monotonic()
    if cmd == _last_cmd and (now - _last_cmd_t) < DEBOUNCE_SEC:
        return False
    _last_cmd = cmd; _last_cmd_t = now
    return True


def _save_debug(audio):
    global _debug_idx
    if not DEBUG_SEGMENTS:
        return
    d = os.path.join(os.path.dirname(__file__), 'debug_segments')
    os.makedirs(d, exist_ok=True)
    _debug_idx += 1
    sf.write(os.path.join(d, f'seg_{_debug_idx:03d}.wav'), audio, SAMPLE_RATE, subtype='PCM_16')


def _add_history(label):
    _history.append(label)
    if len(_history) > MAX_HISTORY:
        _history.pop(0)
    txt_hist.set_text('\n'.join(f'• {c}' for c in reversed(_history)))


def _dispatch(result):
    if result['type'] == 'simple':
        cmd = result['cmd']
        prev_mode = robot.mode
        robot.apply_command(cmd)
        _add_history(cmd)

        if robot.mode != prev_mode:
            if robot.mode == 'auto':
                autopilot.start(robot)
                robot.update_path(grid_map.path_world(autopilot.path))
            else:
                autopilot.stop(robot)
                robot.clear_path()
        elif cmd in ('стоп', 'остановись') and autopilot.active:
            autopilot.stop(robot)
            robot.clear_path()

    elif result['type'] == 'intent':
        s = result['slots']
        if result['intent'] == 'turn':
            robot.rotate_by_degrees(s['direction'] * s['angle'])
        elif result['intent'] == 'move':
            robot.move_by_steps(s['steps'])
        _add_history(result['label'])


def _handle(cmd):
    for r in engine.process(cmd):
        _dispatch(r)


def _vu(rms):
    lv = min(rms / VU_MAX_RMS, 1.0)
    vu_bar[0].set_width(lv)
    vu_bar[0].set_color('#2563eb' if lv < 0.60 else '#d97706' if lv < 0.80 else '#dc2626')


def _wave():
    step = max(1, DISPLAY_SAMPLES // WAVE_POINTS)
    wave_line.set_ydata(_display_buf[::step][:WAVE_POINTS])


def _mel_worker(buf: np.ndarray):
    global _mel_cache, _mel_busy
    try:
        m = extract_log_mel(buf)
        if m.shape[1] < _MEL_FRAMES:
            m = np.pad(m, ((0, 0), (0, _MEL_FRAMES - m.shape[1])), constant_values=-50.0)
        _mel_cache = m[:, :_MEL_FRAMES].astype(np.float32)
    finally:
        _mel_busy = False


def _mel(silence):
    global _mel_busy, _mel_cache
    if silence:
        _mel_cache = np.full((64, _MEL_FRAMES), -50.0, dtype=np.float32)
    elif not _mel_busy:
        _mel_busy = True
        _mel_pool.submit(_mel_worker, _display_buf.copy())
    mel_img.set_data(_mel_cache)


def _flush():
    global _last_redraw
    now = time.monotonic()
    if now - _last_redraw < _GUI_INTERVAL:
        return
    _last_redraw = now
    plt.pause(0.001)


def r_calib(rms):
    _wave(); _vu(rms)
    txt_status.set_text('● Calibrating...'); txt_status.set_color('#d97706')
    txt_cmd.set_text('Mic calibration'); txt_cmd.set_color('#94a3b8')
    txt_conf.set_text('')
    _flush()


def r_record(rms):
    _wave(); _vu(rms)
    txt_status.set_text('● Recording'); txt_status.set_color('#16a34a')
    txt_cmd.set_text('Listening...'); txt_cmd.set_color('#2563eb')
    _flush()


def r_infer(rms):
    _wave(); _vu(rms)
    txt_status.set_text('◎ Processing...'); txt_status.set_color('#7c3aed')
    _flush()


def r_silence(rms):
    _wave(); _vu(rms); _mel(True)
    txt_status.set_text('● Waiting'); txt_status.set_color('#16a34a')
    txt_cmd.set_text('Speak a command...'); txt_cmd.set_color('#cbd5e1')
    txt_conf.set_text('')
    _flush()


def r_result(probs, rms, do_action=True):
    global _result_until, _last_probs
    top_idx = np.argsort(probs)[::-1][:NUM_TOP]
    top_scores = probs[top_idx]
    top_names = [class_names[i] for i in top_idx]
    conf = float(probs.max())
    best = class_names[int(probs.argmax())]
    is_unknown = best == UNKNOWN_CLASS
    recognized = conf >= CONFIDENCE_THRESHOLD and not is_unknown

    if do_action:
        triggered = _should_trigger(best, conf) and not is_unknown
        logger.log(best, conf, worker.last_latency_ms, triggered)
        if triggered:
            _handle(best)

    _result_until = time.monotonic() + RESULT_HOLD_SEC
    _last_probs = probs

    _wave(); _vu(rms)
    if do_action:
        _mel(not recognized)

    for i, (bar, score) in enumerate(zip(prob_bars, top_scores)):
        bar.set_width(score)
        bar.set_color('#16a34a' if (i == 0 and recognized) else
                      '#d97706' if i == 0 else '#3b82f6')
    ax_prob.set_yticklabels(top_names, color='#334155')

    txt_status.set_text('● Waiting'); txt_status.set_color('#16a34a')
    if is_unknown:
        txt_cmd.set_text('Background noise'); txt_cmd.set_color('#94a3b8')
        txt_conf.set_text(f'{conf*100:.1f}%')
    elif not recognized:
        txt_cmd.set_text('Unknown'); txt_cmd.set_color('#dc2626')
        txt_conf.set_text(f'{conf*100:.1f}%')
    else:
        txt_conf.set_text(f'{conf*100:.1f}%')
        label = engine.pending_label if engine.pending_label else best
        color = '#d97706' if engine.pending_label else '#1d4ed8'
        txt_cmd.set_text(label); txt_cmd.set_color(color)
    _flush()


def _audio_cb(indata, *_):
    if not audio_q.full():
        audio_q.put_nowait(indata[:, 0].copy())


def main():
    global _display_buf

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                             blocksize=CHUNK_SAMPLES, callback=_audio_cb, dtype='float32')
    with stream:
        while plt.fignum_exists(fig.number):
            for r in engine.tick():
                _dispatch(r)

            if autopilot.active:
                if not autopilot.tick(robot):
                    robot.clear_path()
                    _add_history('Target reached')

            try:
                chunk = audio_q.get(timeout=0.05)
            except queue.Empty:
                plt.pause(0.01); continue

            _display_buf = np.roll(_display_buf, -len(chunk))
            _display_buf[-len(chunk):] = chunk
            disp_rms = _rms(_display_buf)

            if vad.is_calibrating():
                vad.feed(chunk)
                r_calib(_rms(chunk))
                if not vad.is_calibrating():
                    thr_line.set_xdata([min(vad.threshold / VU_MAX_RMS, 1.0)] * 2)
                continue

            seg = vad.feed(chunk)
            if seg is not None:
                _save_debug(seg)
                if not worker.submit(seg):
                    try:
                        _seg_q.put_nowait(seg)
                    except queue.Full:
                        print('[INF] queue full — dropped')

            probs = worker.poll()
            if probs is not None:
                r_result(probs, disp_rms, do_action=True)
                try:
                    queued = _seg_q.get_nowait()
                    worker.submit(queued)
                except queue.Empty:
                    pass
                continue

            now = time.monotonic()
            if worker.busy():
                if _last_probs is not None and now < _result_until:
                    r_result(_last_probs, disp_rms, do_action=False)
                else:
                    r_infer(disp_rms)
            elif vad.is_recording():
                r_record(disp_rms)
            elif _last_probs is not None and now < _result_until:
                r_result(_last_probs, disp_rms, do_action=False)
            else:
                r_silence(disp_rms)


if __name__ == '__main__':
    try:
        main()
    finally:
        logger.save()
