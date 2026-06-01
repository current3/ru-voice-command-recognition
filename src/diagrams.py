import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

RESULTS = os.path.join(os.path.dirname(__file__), 'results', 'diagrams')
os.makedirs(RESULTS, exist_ok=True)

CP,  CPE = '#dbeafe', '#2563eb'
CD,  CDE = '#fef9c3', '#d97706'
CI,  CIE = '#dcfce7', '#16a34a'
CT,  CTE = '#f1f5f9', '#64748b'
CARR = '#475569'
CTXT = '#1e293b'


def draw_box(ax, cx, cy, w, h, text, fc=CP, ec=CPE, fs=9):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.08',
        facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, color=CTXT, fontweight='bold',
            multialignment='center', zorder=4)


def draw_diamond(ax, cx, cy, w, h, text, fs=9):
    pts = np.array([[cx, cy+h/2], [cx+w/2, cy],
                    [cx, cy-h/2], [cx-w/2, cy]])
    ax.add_patch(Polygon(pts, closed=True,
                         facecolor=CD, edgecolor=CDE,
                         linewidth=1.6, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, color=CTXT, fontweight='bold',
            multialignment='center', zorder=4)


def draw_term(ax, cx, cy, w, h, text, fs=9):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.18',
        facecolor=CT, edgecolor=CTE, linewidth=1.6, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, color=CTXT, fontweight='bold',
            multialignment='center', zorder=4)


def arr(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=CARR,
                                lw=1.4, mutation_scale=14))
    if label:
        mx = (x1+x2)/2 + (0.12 if x2 > x1 else -0.12 if x2 < x1 else 0.12)
        my = (y1+y2)/2
        ax.text(mx, my, label, fontsize=8, color=CDE,
                fontweight='bold', va='center', ha='left')


def larr(ax, x1, y1, xm, y2, label='', label_side='right'):
    ym = y2
    ax.plot([x1, x1], [y1, ym], color=CARR, lw=1.4, zorder=2)
    ax.annotate('', xy=(xm, y2), xytext=(x1, ym),
                arrowprops=dict(arrowstyle='->', color=CARR,
                                lw=1.4, mutation_scale=14))
    if label:
        off = 0.12 if label_side == 'right' else -0.12
        ax.text(x1 + off, (y1 + ym)/2, label, fontsize=8,
                color=CDE, fontweight='bold', va='center')


def save_fig(fig, name):
    path = os.path.join(RESULTS, name)
    fig.savefig(path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')


def diagram_system():
    fig, ax = plt.subplots(figsize=(5, 13))
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(0, 13)
    ax.axis('off'); fig.patch.set_facecolor('white')
    ax.set_title('System Architecture\nVoice Command Recognition',
                 fontsize=12, fontweight='bold', color=CTXT, pad=12)

    W, H = 3.6, 0.76
    blocks = [
        (12.2, 'Microphone',                               CI,  CIE),
        (11.0, 'Audio Capture',                            CP,  CPE),
        ( 9.8, 'Voice Activity Detection (VAD)',           CP,  CPE),
        ( 8.6, 'Signal Preprocessing',                     CP,  CPE),
        ( 7.4, 'Log Mel Spectrogram',                      CP,  CPE),
        ( 6.2, 'Feature Extraction\n(Whisper tiny, frozen)', CP, CPE),
        ( 5.0, 'Temporal Average Pooling\nover speech frames', CP, CPE),
        ( 3.8, 'MLP Classifier',                           CP,  CPE),
        ( 2.6, 'Decision\n(confidence threshold)',          CP,  CPE),
        ( 1.4, 'Intent & Slot Engine',                     CP,  CPE),
        ( 0.3, 'Robot Control',                            CI,  CIE),
    ]

    for y, txt, fc, ec in blocks:
        draw_box(ax, 0, y, W, H, txt, fc=fc, ec=ec)
    for i in range(len(blocks)-1):
        arr(ax, 0, blocks[i][0]-H/2, 0, blocks[i+1][0]+H/2)

    save_fig(fig, '1_system_structure.png')


def diagram_mel():
    fig, ax = plt.subplots(figsize=(5, 11))
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(0, 11)
    ax.axis('off'); fig.patch.set_facecolor('white')
    ax.set_title('Log Mel Spectrogram\nComputation',
                 fontsize=12, fontweight='bold', color=CTXT, pad=12)

    W, H = 3.8, 0.76
    blocks = [
        (10.1, 'Speech signal x[n]',                      CI, CIE),
        ( 8.9, 'Window function (Hamming)',                CP, CPE),
        ( 7.7, 'Short-Time Fourier Transform (STFT)',      CP, CPE),
        ( 6.5, 'Power spectrum\nP(k,m) = |X(k,m)|²',      CP, CPE),
        ( 5.3, 'Mel filter bank (80 bands)',               CP, CPE),
        ( 4.1, 'Mel energies E_m(τ)',                      CP, CPE),
        ( 2.9, 'Log compression\nM_m(τ) = log(E_m(τ) + ε)', CP, CPE),
        ( 1.5, 'Log Mel Spectrogram (80 × 3000)',          CI, CIE),
    ]

    for y, txt, fc, ec in blocks:
        draw_box(ax, 0, y, W, H, txt, fc=fc, ec=ec)
    for i in range(len(blocks)-1):
        arr(ax, 0, blocks[i][0]-H/2, 0, blocks[i+1][0]+H/2)

    save_fig(fig, '2_mel_spectrogram.png')


def diagram_recognition():
    fig, ax = plt.subplots(figsize=(7, 13))
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(0, 13)
    ax.axis('off'); fig.patch.set_facecolor('white')
    ax.set_title('Command Recognition Algorithm',
                 fontsize=12, fontweight='bold', color=CTXT, pad=12)

    W, H   = 3.8, 0.76
    DW, DH = 3.0, 0.90

    draw_term   (ax,  0, 12.4, 2.6, 0.65, 'Start')
    draw_box    (ax,  0, 11.4, W,   H,    'Speech segment (audio signal)')
    draw_box    (ax,  0, 10.3, W,   H,    'Log Mel Spectrogram')
    draw_box    (ax,  0,  9.2, W,   H,    'Feature extraction\n(Whisper tiny)')
    draw_box    (ax,  0,  8.1, W,   H,    'Temporal pooling → embedding z')
    draw_box    (ax,  0,  7.0, W,   H,    'MLP classifier\n→ probability vector p')
    draw_box    (ax,  0,  5.9, W,   H,    'k* = argmax p_k\nconf = max p_k')
    draw_diamond(ax,  0,  4.6, DW,  DH,   'conf > θ ?')
    draw_box    (ax,  0,  3.3, W,   H,    'Command k* → intent engine')
    draw_box    (ax, 2.8, 4.6, 2.2, H,    'Unknown\nclass')
    draw_term   (ax,  0,  2.0, 2.6, 0.65, 'End')

    for y1, y2 in [(12.4-0.33, 11.4+H/2), (11.4-H/2, 10.3+H/2),
                   (10.3-H/2,  9.2+H/2),  ( 9.2-H/2,  8.1+H/2),
                   ( 8.1-H/2,  7.0+H/2),  ( 7.0-H/2,  5.9+H/2),
                   ( 5.9-H/2,  4.6+DH/2)]:
        arr(ax, 0, y1, 0, y2)

    arr(ax, 0, 4.6-DH/2, 0, 3.3+H/2, 'Yes')
    arr(ax, 0, 3.3-H/2,  0, 2.0+0.33)
    arr(ax, DW/2, 4.6, 1.7, 4.6, 'No')

    ax.plot([2.8, 2.8], [4.6-H/2, 2.0], color=CARR, lw=1.4, zorder=2)
    arr(ax, 2.8, 2.0, 0+1.3, 2.0)

    save_fig(fig, '3_recognition_algorithm.png')


def diagram_realtime():
    fig, ax = plt.subplots(figsize=(8, 14))
    ax.set_xlim(-4, 4); ax.set_ylim(0, 14)
    ax.axis('off'); fig.patch.set_facecolor('white')
    ax.set_title('Real-Time Processing Algorithm',
                 fontsize=12, fontweight='bold', color=CTXT, pad=12)

    W, H   = 4.0, 0.76
    DW, DH = 3.2, 0.90
    XR     = 3.1

    draw_term   (ax,  0, 13.3, 2.8, 0.65, 'Start')
    draw_box    (ax,  0, 12.3, W,   H,    'Capture audio block (100 ms, 16 kHz)')
    draw_diamond(ax,  0, 11.1, DW,  DH,   'Calibration\ncomplete?')
    draw_box    (ax, XR, 11.1, 2.4, H,    'VAD calibration')
    draw_diamond(ax,  0,  9.8, DW,  DH,   'RMS > threshold?')
    draw_box    (ax, XR,  9.8, 2.4, H,    'Silence\n(pre-roll buffer)')
    draw_box    (ax,  0,  8.5, W,   H,    'Accumulate speech segment')
    draw_diamond(ax,  0,  7.2, DW,  DH,   'End of segment\n(silence / limit)?')
    draw_box    (ax,  0,  6.0, W,   H,    'Feature extraction\n→ classification')
    draw_diamond(ax,  0,  4.8, DW,  DH,   'conf > θ ?')
    draw_box    (ax, XR,  4.8, 2.4, H,    'Unknown\nclass')
    draw_box    (ax,  0,  3.5, W,   H,    'Execute command\nUpdate UI')
    draw_term   (ax,  0,  2.4, 2.8, 0.65, 'Repeat')

    arr(ax, 0, 13.3-0.33, 0, 12.3+H/2)
    arr(ax, 0, 12.3-H/2,  0, 11.1+DH/2)
    arr(ax, 0, 11.1-DH/2, 0, 9.8+DH/2,  'Yes')
    arr(ax, 0,  9.8-DH/2, 0, 8.5+H/2,   'Yes')
    arr(ax, 0,  8.5-H/2,  0, 7.2+DH/2)
    arr(ax, 0,  7.2-DH/2, 0, 6.0+H/2,   'Yes')
    arr(ax, 0,  6.0-H/2,  0, 4.8+DH/2)
    arr(ax, 0,  4.8-DH/2, 0, 3.5+H/2,   'Yes')
    arr(ax, 0,  3.5-H/2,  0, 2.4+0.33)

    arr(ax, DW/2, 11.1, XR-1.2, 11.1, 'No')
    arr(ax, DW/2,  9.8, XR-1.2,  9.8, 'No')
    arr(ax, DW/2,  4.8, XR-1.2,  4.8, 'No')

    XL_loop = -3.5
    ax.plot([-DW/2, XL_loop], [7.2, 7.2], color=CARR, lw=1.4, zorder=2)
    ax.plot([XL_loop, XL_loop], [7.2, 8.5], color=CARR, lw=1.4, zorder=2)
    ax.annotate('', xy=(-W/2, 8.5), xytext=(XL_loop, 8.5),
                arrowprops=dict(arrowstyle='->', color=CARR, lw=1.4, mutation_scale=14))
    ax.text(-DW/2-0.15, 7.2+0.12, 'No', fontsize=8, color=CDE,
            fontweight='bold', ha='right')

    ax.plot([XR, XR], [11.1+H/2, 12.7], color=CARR, lw=1.4, zorder=2)
    ax.plot([XR, 0+W/2], [12.7, 12.7], color=CARR, lw=1.4, zorder=2)
    ax.annotate('', xy=(W/2, 12.3+H/2), xytext=(W/2, 12.7),
                arrowprops=dict(arrowstyle='->', color=CARR, lw=1.4, mutation_scale=14))

    ax.plot([XR, XR], [9.8+H/2, 12.7+0.3], color=CARR, lw=1.4, zorder=2, linestyle='dashed')

    ax.plot([XR, XR], [4.8-H/2, 3.5], color=CARR, lw=1.4, zorder=2)
    arr(ax, XR, 3.5, W/2, 3.5)

    ax.plot([-3.0, -3.0], [2.4, 12.3], color='#94a3b8', lw=1.3, linestyle='dashed', zorder=2)
    ax.plot([-3.0, -W/2], [12.3, 12.3], color='#94a3b8', lw=1.3, linestyle='dashed', zorder=2)
    ax.plot([-3.0, -W/2-0.2], [2.4, 2.4], color='#94a3b8', lw=1.3, linestyle='dashed', zorder=2)
    ax.annotate('', xy=(-W/2, 12.3), xytext=(-W/2-0.1, 12.3),
                arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1.3, mutation_scale=12))
    ax.text(-3.2, 7.5, 'Loop', fontsize=8, color='#94a3b8',
            rotation=90, va='center', ha='center')

    save_fig(fig, '4_realtime_algorithm.png')


def diagram_intent():
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(-6.5, 6.5); ax.set_ylim(0, 14)
    ax.axis('off'); fig.patch.set_facecolor('white')
    ax.set_title('Intent & Slot Engine Algorithm',
                 fontsize=12, fontweight='bold', color=CTXT, pad=12)

    W, H   = 3.4, 0.76
    DW, DH = 2.8, 0.85
    XLC = -2.0
    XRC =  2.0
    XL  = -5.1
    XR  =  5.1
    WS  =  2.0

    draw_term   (ax,  0, 13.3, 3.2, 0.65, 'Command recognized')
    draw_diamond(ax,  0, 12.1, DW,  DH,   'Active\nintent?')

    draw_diamond(ax, XLC, 10.8, DW, DH,  'Timeout\nexpired?')
    draw_box    (ax,  XL, 10.8, WS,  H,  'Execute with\ndefaults (or cancel)')
    draw_diamond(ax, XLC,  9.4, DW, DH,  'Command fills\na slot?')
    draw_box    (ax,  XL,  9.4, WS,  H,  'Execute current\n→ process separately')
    draw_diamond(ax, XLC,  8.0, DW, DH,  'All slots\nfilled?')
    draw_box    (ax,  XL,  8.0, WS,  H,  'Waiting for\nnext slot')

    draw_diamond(ax, XRC, 10.8, DW, DH,  'Trigger\ncommand?')
    draw_box    (ax,  XR, 10.8, WS,  H,  'Simple command\n→ execute')
    draw_box    (ax, XRC,  9.4,  W,  H,  'Activate intent\nStart timeout')

    draw_box    (ax,  0,  6.6,  W,  H,   'Execute intent')
    draw_term   (ax,  0,  5.3, 4.2, 0.65,'Waiting for next command')

    arr(ax, 0, 13.3-0.33, 0, 12.1+DH/2)

    ax.plot([-DW/2, XLC], [12.1, 12.1], color=CARR, lw=1.4, zorder=2)
    ax.annotate('', xy=(XLC, 10.8+DH/2), xytext=(XLC, 12.1),
                arrowprops=dict(arrowstyle='->', color=CARR, lw=1.4, mutation_scale=14))
    ax.text(-DW/2-0.15, 12.2, 'Yes', fontsize=8, color=CDE, fontweight='bold', ha='right')

    ax.plot([DW/2, XRC], [12.1, 12.1], color=CARR, lw=1.4, zorder=2)
    ax.annotate('', xy=(XRC, 10.8+DH/2), xytext=(XRC, 12.1),
                arrowprops=dict(arrowstyle='->', color=CARR, lw=1.4, mutation_scale=14))
    ax.text(DW/2+0.15, 12.2, 'No', fontsize=8, color=CDE, fontweight='bold')

    ax.annotate('', xy=(XL+WS/2, 10.8), xytext=(XLC-DW/2, 10.8),
                arrowprops=dict(arrowstyle='->', color=CARR, lw=1.4, mutation_scale=14))
    ax.text((XLC-DW/2+XL+WS/2)/2, 10.8+0.13, 'Yes',
            fontsize=8, color=CDE, fontweight='bold', ha='center')

    arr(ax, XLC, 10.8-DH/2, XLC, 9.4+DH/2, 'No')

    ax.annotate('', xy=(XL+WS/2, 9.4), xytext=(XLC-DW/2, 9.4),
                arrowprops=dict(arrowstyle='->', color=CARR, lw=1.4, mutation_scale=14))
    ax.text((XLC-DW/2+XL+WS/2)/2, 9.4+0.13, 'No',
            fontsize=8, color=CDE, fontweight='bold', ha='center')

    arr(ax, XLC, 9.4-DH/2, XLC, 8.0+DH/2, 'Yes')

    ax.annotate('', xy=(XL+WS/2, 8.0), xytext=(XLC-DW/2, 8.0),
                arrowprops=dict(arrowstyle='->', color=CARR, lw=1.4, mutation_scale=14))
    ax.text((XLC-DW/2+XL+WS/2)/2, 8.0+0.13, 'No',
            fontsize=8, color=CDE, fontweight='bold', ha='center')

    ax.plot([XLC, XLC], [8.0-DH/2, 6.6], color=CARR, lw=1.4, zorder=2)
    ax.annotate('', xy=(-W/2, 6.6), xytext=(XLC, 6.6),
                arrowprops=dict(arrowstyle='->', color=CARR, lw=1.4, mutation_scale=14))
    ax.text(XLC+0.12, 8.0-DH/2-0.16, 'Yes', fontsize=8, color=CDE, fontweight='bold')

    ax.annotate('', xy=(XR-WS/2, 10.8), xytext=(XRC+DW/2, 10.8),
                arrowprops=dict(arrowstyle='->', color=CARR, lw=1.4, mutation_scale=14))
    ax.text((XRC+DW/2+XR-WS/2)/2, 10.8+0.13, 'No',
            fontsize=8, color=CDE, fontweight='bold', ha='center')

    arr(ax, XRC, 10.8-DH/2, XRC, 9.4+H/2, 'Yes')

    ax.plot([XRC, XRC], [9.4-H/2, 5.3], color=CARR, lw=1.4, zorder=2)
    ax.annotate('', xy=(W/2+0.15, 5.3), xytext=(XRC, 5.3),
                arrowprops=dict(arrowstyle='->', color=CARR, lw=1.4, mutation_scale=14))

    ax.plot([XR, XR], [10.8-H/2, 5.3], color=CARR, lw=1.4, linestyle='dashed', alpha=0.7, zorder=2)
    ax.plot([XR, W/2+0.2], [5.3, 5.3], color=CARR, lw=1.4, linestyle='dashed', alpha=0.7, zorder=2)

    ax.plot([XL, XL], [10.8-H/2, 5.3], color=CARR, lw=1.4, linestyle='dashed', alpha=0.7, zorder=2)
    ax.plot([XL, -W/2-0.15], [5.3, 5.3], color=CARR, lw=1.4, linestyle='dashed', alpha=0.7, zorder=2)

    XLL = -6.1
    ax.plot([XL-WS/2, XLL], [9.4, 9.4], color='#94a3b8', lw=1.2, linestyle='dotted', zorder=2)
    ax.plot([XL-WS/2, XLL], [8.0, 8.0], color='#94a3b8', lw=1.2, linestyle='dotted', zorder=2)
    ax.plot([XLL, XLL], [8.0, 13.3], color='#94a3b8', lw=1.2, linestyle='dotted', zorder=2)
    ax.annotate('', xy=(-W/2+0.4, 13.3), xytext=(XLL, 13.3),
                arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1.2, mutation_scale=12))
    ax.text(XLL-0.22, 10.8, 'Loop', fontsize=8, color='#94a3b8',
            rotation=90, va='center', ha='center')

    arr(ax, 0, 6.6-H/2, 0, 5.3+0.33)

    save_fig(fig, '5_intent_engine.png')


if __name__ == '__main__':
    print('Generating diagrams...')
    diagram_system()
    diagram_mel()
    diagram_recognition()
    diagram_realtime()
    diagram_intent()
    print('\nDone. Files saved to results/diagrams/')
