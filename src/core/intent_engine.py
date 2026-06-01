import time
from config import SLOT_TYPES, INTENT_TRIGGERS, INTENTS


class IntentEngine:

    def __init__(self):
        self._active: str | None = None
        self._slots: dict = {}
        self._label: str = ''
        self._deadline: float = 0.0
        self.pending_label: str = ''

    def process(self, cmd: str) -> list[dict]:
        now = time.monotonic()

        if self._active and now > self._deadline:
            expired = self._try_execute()
            self._reset()
            results = [expired] if expired else [{'type': 'cancel'}]
            results += self.process(cmd)
            return results

        if cmd in INTENT_TRIGGERS:
            results = []
            if self._active:
                prev = self._try_execute()
                if prev:
                    results.append(prev)
                self._reset()
            self._start(cmd)
            return results

        if self._active:
            if self._fill_slot(cmd):
                if self._all_slots_filled():
                    result = self._try_execute()
                    self._reset()
                    return [result] if result else []
                return []
            else:
                results = []
                prev = self._try_execute()
                self._reset()
                if prev:
                    results.append(prev)
                results += self.process(cmd)
                return results

        return [{'type': 'simple', 'cmd': cmd}]

    def tick(self) -> list[dict]:
        if not self._active:
            return []
        if time.monotonic() > self._deadline:
            result = self._try_execute()
            self._reset()
            return [result] if result else [{'type': 'cancel'}]
        return []

    def _start(self, trigger_cmd: str):
        intent_name, label = INTENT_TRIGGERS[trigger_cmd]
        self._active = intent_name
        self._slots = {}
        self._label = label
        self._deadline = time.monotonic() + INTENTS[intent_name]['timeout']
        self.pending_label = f'{label} ?'

    def _fill_slot(self, cmd: str) -> bool:
        intent = INTENTS[self._active]
        for slot_def in intent['slots']:
            name = slot_def['name']
            if name in self._slots:
                continue
            slot_values = SLOT_TYPES[slot_def['type']]
            if cmd in slot_values:
                self._slots[name] = {'value': slot_values[cmd], 'word': cmd}
                self._update_pending_label()
                return True
        return False

    def _can_execute(self) -> bool:
        intent = INTENTS[self._active]
        return all(
            s['name'] in self._slots
            for s in intent['slots'] if s['required']
        )

    def _all_slots_filled(self) -> bool:
        intent = INTENTS[self._active]
        return all(s['name'] in self._slots for s in intent['slots'])

    def _try_execute(self) -> dict | None:
        if not self._can_execute():
            return {'type': 'cancel'}

        intent = INTENTS[self._active]
        slots_out = {}
        for slot_def in intent['slots']:
            name = slot_def['name']
            slots_out[name] = (
                self._slots[name]['value'] if name in self._slots else slot_def['default']
            )

        return {
            'type':   'intent',
            'intent': self._active,
            'slots':  slots_out,
            'label':  self._build_label(),
        }

    def _build_label(self) -> str:
        label = self._label
        for slot_def in INTENTS[self._active]['slots']:
            name = slot_def['name']
            if name in self._slots:
                label += f" {self._slots[name]['word'].upper()}"
        return label

    def _update_pending_label(self):
        filled = self._build_label()
        intent = INTENTS[self._active]
        unfilled = sum(1 for s in intent['slots'] if s['name'] not in self._slots)
        self.pending_label = f'{filled} ?' if unfilled else filled

    def _reset(self):
        self._active = None
        self._slots = {}
        self._label = ''
        self._deadline = 0.0
        self.pending_label = ''
