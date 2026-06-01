import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import LineCollection

from config import COMMAND_ACTIONS, MAP_LIMIT

MAX_TRAIL = 80


class RobotVisualizer:
    MAP_LIMIT = MAP_LIMIT
    SPEED_STEP = 2.0
    SPEED_LEVELS = {1: 1.0, 2: 2.0, 3: 4.0}

    _BODY = np.array([
        [-1.2, -1.0],
        [ 0.4, -1.0],
        [ 1.4,  0.0],
        [ 0.4,  1.0],
        [-1.2,  1.0],
    ])
    _WHEEL_L = np.array([[-1.0, -1.45], [ 0.7, -1.45],
                          [ 0.7, -1.05], [-1.0, -1.05]])
    _WHEEL_R = np.array([[-1.0,  1.05], [ 0.7,  1.05],
                          [ 0.7,  1.45], [-1.0,  1.45]])
    _ANTENNA = np.array([[0.3, 0.0], [1.8, 0.0]])

    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self.x = 0.0
        self.y = 0.0
        self.angle = 90.0
        self.speed = 2.0
        self.powered = True
        self.mode = 'manual'
        self.trail_x = [0.0]
        self.trail_y = [0.0]

        self._setup_axes()
        self._init_artists()
        self._draw()

    def _world(self, local: np.ndarray) -> np.ndarray:
        rad = np.radians(self.angle)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]])
        return local @ R.T + np.array([self.x, self.y])

    def _path_length(self) -> float:
        dx = np.diff(self.trail_x)
        dy = np.diff(self.trail_y)
        return float(np.sum(np.sqrt(dx**2 + dy**2)))

    def _setup_axes(self):
        lim = self.MAP_LIMIT
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_facecolor('#f0f7ff')
        self.ax.set_aspect('equal')
        self.ax.set_title('Robot Navigation Map', color='#1e293b',
                          fontsize=11, fontweight='bold', pad=8)
        self.ax.tick_params(colors='#94a3b8', labelsize=7)
        for sp in self.ax.spines.values():
            sp.set_edgecolor('#bfdbfe')

        for v in range(-lim, lim + 1, 4):
            self.ax.axhline(v, color='#dbeafe', linewidth=0.5, zorder=0)
            self.ax.axvline(v, color='#dbeafe', linewidth=0.5, zorder=0)

        self.ax.axhline(0, color='#bfdbfe', linewidth=1.0, zorder=0)
        self.ax.axvline(0, color='#bfdbfe', linewidth=1.0, zorder=0)

        self.ax.plot(0, 0, '*', color='#f59e0b', markersize=12, zorder=3)

        offset = lim - 1.0
        for txt, x, y in [('N', 0, offset), ('S', 0, -offset),
                           ('E', offset, 0),  ('W', -offset, 0)]:
            self.ax.text(x, y, txt, color='#94a3b8', fontsize=8,
                         ha='center', va='center', fontweight='bold')

    def _init_artists(self):
        self._trail = LineCollection([], linewidths=2.0, zorder=2)
        self.ax.add_collection(self._trail)

        self._waypts, = self.ax.plot([], [], 'o', markersize=3,
                                     color='#4A90D9', alpha=0.45, zorder=2)

        self._wheel_l = Polygon(self._world(self._WHEEL_L), closed=True,
                                facecolor='#64748b', edgecolor='#475569',
                                linewidth=1, zorder=4)
        self._wheel_r = Polygon(self._world(self._WHEEL_R), closed=True,
                                facecolor='#64748b', edgecolor='#475569',
                                linewidth=1, zorder=4)
        self.ax.add_patch(self._wheel_l)
        self.ax.add_patch(self._wheel_r)

        self._body = Polygon(self._world(self._BODY), closed=True,
                             facecolor='#2563eb', edgecolor='#93c5fd',
                             linewidth=1.5, zorder=5)
        self.ax.add_patch(self._body)

        ant = self._world(self._ANTENNA)
        self._antenna, = self.ax.plot(ant[:, 0], ant[:, 1],
                                      color='#f59e0b', linewidth=1.5, zorder=6)
        self._antenna_tip, = self.ax.plot([ant[-1, 0]], [ant[-1, 1]],
                                          'o', color='#f59e0b', markersize=4, zorder=7)

        self._coord = self.ax.text(
            self.x, self.y - 3.0, '',
            color='#64748b', fontsize=7, ha='center', zorder=6
        )

        self._info = self.ax.text(
            -self.MAP_LIMIT + 0.6, self.MAP_LIMIT - 0.6, '',
            fontsize=8.5, color='#1e293b', verticalalignment='top',
            linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='white', edgecolor='#bfdbfe', alpha=0.95),
            zorder=8
        )

    def _draw(self):
        if self.powered:
            body_color = '#2563eb' if self.mode == 'manual' else '#7c3aed'
            edge_color = '#93c5fd' if self.mode == 'manual' else '#c4b5fd'
        else:
            body_color, edge_color = '#dc2626', '#fca5a5'

        self._body.set_xy(self._world(self._BODY))
        self._body.set_facecolor(body_color)
        self._body.set_edgecolor(edge_color)

        self._wheel_l.set_xy(self._world(self._WHEEL_L))
        self._wheel_r.set_xy(self._world(self._WHEEL_R))

        ant = self._world(self._ANTENNA)
        self._antenna.set_xdata(ant[:, 0])
        self._antenna.set_ydata(ant[:, 1])
        self._antenna_tip.set_xdata([ant[-1, 0]])
        self._antenna_tip.set_ydata([ant[-1, 1]])

        self._coord.set_position((self.x, self.y - 3.0))
        self._coord.set_text(f'({self.x:.1f}, {self.y:.1f})')

        pts = np.column_stack([self.trail_x, self.trail_y])
        if len(pts) >= 2:
            segs = [pts[i:i+2] for i in range(len(pts) - 1)]
            n = len(segs)
            colors = [(*plt.cm.Blues(0.3 + 0.7 * (i / max(n, 1)))[:3],
                       0.2 + 0.75 * (i / max(n, 1))) for i in range(n)]
            self._trail.set_segments(segs)
            self._trail.set_color(colors)

        self._waypts.set_xdata(self.trail_x)
        self._waypts.set_ydata(self.trail_y)

        mode_str = 'AUTO' if self.mode == 'auto' else 'MANUAL'
        power_str = '● ON' if self.powered else '○ OFF'
        dist = self._path_length()
        self._info.set_text(
            f'Speed  : {self.speed:.0f}\n'
            f'Heading: {self.angle:.0f}°\n'
            f'Mode   : {mode_str}\n'
            f'Power  : {power_str}\n'
            f'Path   : {dist:.1f} u'
        )

    def rotate_by_degrees(self, degrees: float):
        self.angle = (self.angle + degrees) % 360
        self._draw()

    def move_by_steps(self, steps: int):
        if not self.powered:
            return
        rad = np.radians(self.angle)
        self.x = float(np.clip(self.x + np.cos(rad) * self.speed * steps,
                               -self.MAP_LIMIT, self.MAP_LIMIT))
        self.y = float(np.clip(self.y + np.sin(rad) * self.speed * steps,
                               -self.MAP_LIMIT, self.MAP_LIMIT))
        self._append_trail()
        self._draw()

    def apply_command(self, command: str):
        if command not in COMMAND_ACTIONS:
            return
        action, value = COMMAND_ACTIONS[command]

        if not self.powered and action in ('move', 'rotate'):
            return

        if action == 'move':
            rad = np.radians(self.angle)
            self.x = float(np.clip(
                self.x + np.cos(rad) * self.speed * value,
                -self.MAP_LIMIT, self.MAP_LIMIT))
            self.y = float(np.clip(
                self.y + np.sin(rad) * self.speed * value,
                -self.MAP_LIMIT, self.MAP_LIMIT))
            self._append_trail()
        elif action == 'rotate':
            self.angle = (self.angle + value) % 360
        elif action == 'stop':
            pass
        elif action == 'speed_up':
            self.speed = min(self.speed + self.SPEED_STEP, 8.0)
        elif action == 'speed_down':
            self.speed = max(self.speed - self.SPEED_STEP, 1.0)
        elif action == 'set_speed':
            self.speed = self.SPEED_LEVELS.get(value, self.speed)
        elif action == 'home':
            self.x, self.y, self.angle = 0.0, 0.0, 90.0
            self.trail_x, self.trail_y = [0.0], [0.0]
        elif action == 'power':
            self.powered = value
        elif action == 'mode':
            self.mode = value
        elif action == 'toggle_mode':
            self.mode = 'auto' if self.mode == 'manual' else 'manual'

        self._draw()

    def _append_trail(self):
        self.trail_x.append(self.x)
        self.trail_y.append(self.y)
        if len(self.trail_x) > MAX_TRAIL:
            self.trail_x = self.trail_x[-MAX_TRAIL:]
            self.trail_y = self.trail_y[-MAX_TRAIL:]

    def setup_navigation(self, grid_map):
        for x, y, w, h in grid_map.obstacles_world():
            self.ax.add_patch(Rectangle(
                (x, y), w, h,
                facecolor='#c0392b', edgecolor='#e74c3c',
                alpha=0.75, linewidth=1, zorder=1,
            ))

        tx, ty = grid_map.target_world
        self.ax.plot(tx, ty, 'D', color='#f39c12', markersize=12,
                     markeredgecolor='#e67e22', markeredgewidth=1.5, zorder=3)
        self.ax.text(tx, ty - 2.5, 'B', color='#f39c12',
                     fontsize=9, ha='center', fontweight='bold', zorder=3)

        self._path_line, = self.ax.plot([], [], '--',
                                        color='#9b59b6', linewidth=1.5,
                                        alpha=0.7, zorder=2)

    def update_path(self, path_world: list):
        if not hasattr(self, '_path_line') or not path_world:
            return
        xs = [p[0] for p in path_world]
        ys = [p[1] for p in path_world]
        self._path_line.set_xdata(xs)
        self._path_line.set_ydata(ys)

    def clear_path(self):
        if hasattr(self, '_path_line'):
            self._path_line.set_xdata([])
            self._path_line.set_ydata([])
