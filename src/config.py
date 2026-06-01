import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'classifier.keras')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.npy')

WHISPER_MODEL = 'openai/whisper-tiny'
EMBEDDING_DIM = 384

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.002

CONFIDENCE_THRESHOLD = 0.18
UNKNOWN_CLASS = 'неизвестно'
CONFIDENCE_OVERRIDES = {
    'автоматический': 0.60,
}

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 5e-5

TEST_SPEAKERS = ['grisha', 'fedos', 'yana', 'nastya', 'zhenya']

MAP_LIMIT = 20
GRID_CELLS = 20
CELL_SIZE = 2
AUTO_STEP_SEC = 0.5
MAP_TARGET = (14.0, 14.0)

MAP_OBSTACLES = [
    (3, 8), (4, 8), (5, 8), (5, 9), (5, 10),
    (8, 13), (9, 13), (10, 13), (10, 14),
    (14, 6), (14, 7), (14, 8), (13, 8),
    (7, 4), (8, 4), (9, 4),
]

COMMAND_ACTIONS = {
    'вперед':         ('move',        1),
    'двигайся':       ('move',        1),
    'назад':          ('move',       -1),
    'вернись':        ('move',       -1),
    'влево':          ('rotate',    +15),
    'налево':         ('rotate',    +15),
    'вправо':         ('rotate',    -15),
    'направо':        ('rotate',    -15),
    'стоп':           ('stop',        0),
    'остановись':     ('stop',        0),
    'быстрее':        ('speed_up',    0),
    'медленнее':      ('speed_down',  0),
    'развернись':     ('rotate',    180),
    'разворот':       ('rotate',    180),
    'домой':          ('home',        0),
    'включить':       ('power',    True),
    'выключить':      ('power',   False),
    'автоматический': ('mode',    'auto'),
    'ручной':         ('mode',  'manual'),
    'режим':          ('toggle_mode',  0),
    'один':           ('set_speed',    1),
    'два':            ('set_speed',    2),
    'три':            ('set_speed',    3),
}

SLOT_TYPES = {
    'direction': {'налево': +1, 'направо': -1, 'влево': +1, 'вправо': -1},
    'angle':     {'один': 30, 'два': 60, 'три': 90},
    'steps':     {'один':  1, 'два':  2, 'три':  3},
}

INTENT_TRIGGERS = {
    'поверни':  ('turn',  'ПОВЕРНИ'),
    'двигайся': ('move',  'ДВИГАЙСЯ'),
}

INTENTS = {
    'turn': {
        'label':   'ПОВЕРНИ',
        'timeout': 4.0,
        'slots': [
            {'name': 'direction', 'type': 'direction', 'required': True,  'default': None},
            {'name': 'angle',     'type': 'angle',     'required': False, 'default': 30},
        ],
    },
    'move': {
        'label':   'ДВИГАЙСЯ',
        'timeout': 2.5,
        'slots': [
            {'name': 'steps', 'type': 'steps', 'required': False, 'default': 1},
        ],
    },
}
