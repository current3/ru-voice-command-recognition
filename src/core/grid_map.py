import heapq
from config import MAP_LIMIT, GRID_CELLS, CELL_SIZE, MAP_OBSTACLES, MAP_TARGET


class GridMap:
    def __init__(self):
        self.size = GRID_CELLS
        self.cell = CELL_SIZE
        self.obstacles = set(map(tuple, MAP_OBSTACLES))
        self.target_world = MAP_TARGET
        self.target_grid = self.world_to_grid(*MAP_TARGET)

    def world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        gx = int((wx + MAP_LIMIT) / self.cell)
        gy = int((wy + MAP_LIMIT) / self.cell)
        return (
            max(0, min(self.size - 1, gx)),
            max(0, min(self.size - 1, gy)),
        )

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        wx = gx * self.cell - MAP_LIMIT + self.cell / 2
        wy = gy * self.cell - MAP_LIMIT + self.cell / 2
        return (wx, wy)

    def astar(self, start_world: tuple) -> list[tuple[int, int]]:
        start = self.world_to_grid(*start_world)
        goal = self.target_grid

        if start == goal:
            return [start]

        open_set = [(0, start)]
        came_from: dict = {}
        g: dict = {start: 0}

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur == goal:
                path = []
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append(start)
                return path[::-1]

            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nb[0] < self.size and 0 <= nb[1] < self.size):
                    continue
                if nb in self.obstacles:
                    continue
                tg = g[cur] + 1
                if tg < g.get(nb, float('inf')):
                    came_from[nb] = cur
                    g[nb] = tg
                    f = tg + abs(nb[0] - goal[0]) + abs(nb[1] - goal[1])
                    heapq.heappush(open_set, (f, nb))

        return []

    def obstacles_world(self) -> list[tuple[float, float, float, float]]:
        rects = []
        for gx, gy in self.obstacles:
            x = gx * self.cell - MAP_LIMIT
            y = gy * self.cell - MAP_LIMIT
            rects.append((x, y, self.cell, self.cell))
        return rects

    def path_world(self, path: list[tuple[int, int]]) -> list[tuple[float, float]]:
        return [self.grid_to_world(gx, gy) for gx, gy in path]
