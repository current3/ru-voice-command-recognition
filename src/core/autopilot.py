import time
import math
from config import CELL_SIZE, AUTO_STEP_SEC


class AutoPilot:
    def __init__(self, grid_map):
        self._map = grid_map
        self._path: list = []
        self._idx = 0
        self._active = False
        self._next_t = 0.0
        self._prev_speed = CELL_SIZE

    def start(self, robot) -> list:
        path = self._map.astar((robot.x, robot.y))
        if len(path) < 2:
            return []
        self._path = path
        self._idx = 0
        self._active = True
        self._next_t = time.monotonic() + AUTO_STEP_SEC
        self._prev_speed = robot.speed
        robot.speed = CELL_SIZE
        return self._map.path_world(path)

    def tick(self, robot) -> bool:
        if not self._active:
            return False
        if time.monotonic() < self._next_t:
            return True
        if self._idx >= len(self._path) - 1:
            self._finish(robot)
            return False

        cur = self._path[self._idx]
        nxt = self._path[self._idx + 1]
        dx, dy = nxt[0] - cur[0], nxt[1] - cur[1]

        required = math.degrees(math.atan2(dy, dx))
        diff = (required - robot.angle + 180) % 360 - 180

        if abs(diff) > 2:
            robot.rotate_by_degrees(diff)

        robot.move_by_steps(1)
        self._idx += 1

        if self._idx >= len(self._path) - 1:
            self._finish(robot)
            return False

        self._next_t = time.monotonic() + AUTO_STEP_SEC
        return True

    def stop(self, robot=None):
        self._active = False
        if robot is not None:
            robot.speed = self._prev_speed

    @property
    def active(self) -> bool:
        return self._active

    @property
    def path(self) -> list:
        return self._path

    @property
    def path_idx(self) -> int:
        return self._idx

    def _finish(self, robot):
        self._active = False
        robot.speed = self._prev_speed
