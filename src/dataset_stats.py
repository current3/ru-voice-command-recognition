import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results')

GROUPS = {
    'Motion':   ['вперед', 'назад', 'двигайся', 'вернись'],
    'Rotation': ['влево', 'вправо', 'налево', 'направо', 'поверни', 'развернись', 'разворот'],
    'Stop':     ['стоп', 'остановись'],
    'Speed':    ['быстрее', 'медленнее', 'один', 'два', 'три'],
    'Control':  ['включить', 'выключить', 'режим', 'автоматический', 'ручной', 'домой'],
}

GROUP_COLORS = {
    'Motion':   '#2563eb',
    'Rotation': '#7c3aed',
    'Stop':     '#dc2626',
    'Speed':    '#d97706',
    'Control':  '#16a34a',
}


def count_files(dataset_path):
    counts = {}
    for class_dir in os.listdir(dataset_path):
        full = os.path.join(dataset_path, class_dir)
        if os.path.isdir(full):
            counts[class_dir] = len([
                f for f in os.listdir(full)
                if os.path.splitext(f)[1].lower() in ('.ogg', '.wav', '.mp3', '.flac')
            ])
    return counts


def main():
    counts = count_files(DATASET_PATH)

    classes, values, colors = [], [], []
    for group, cmds in GROUPS.items():
        for cmd in cmds:
            if cmd in counts:
                classes.append(cmd)
                values.append(counts[cmd])
                colors.append(GROUP_COLORS[group])

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars = ax.bar(classes, values, color=colors, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                str(val), ha='center', va='bottom', fontsize=8, color='#334155')

    legend_elements = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor='#e2e8f0')

    ax.set_title('Dataset Class Distribution', fontsize=13,
                 fontweight='bold', color='#1e293b', pad=12)
    ax.set_xlabel('Command Class', fontsize=10, color='#475569')
    ax.set_ylabel('Recording Count', fontsize=10, color='#475569')
    ax.set_ylim(0, max(values) + 3)
    ax.tick_params(axis='x', rotation=45, labelsize=9, colors='#334155')
    ax.tick_params(axis='y', labelsize=9, colors='#334155')
    for sp in ax.spines.values():
        sp.set_color('#e2e8f0')
    ax.yaxis.grid(True, color='#f1f5f9', linewidth=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(RESULTS_PATH, exist_ok=True)
    out = os.path.join(RESULTS_PATH, 'dataset_distribution.png')
    plt.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
    print(f'Saved: {out}')
    plt.show()


if __name__ == '__main__':
    main()
