from enum import Enum
from random import choice

import pygame
import numpy as np

# 游戏参数
SIZE = 45
GRID_SIZE = 720 // SIZE
WIDTH = SIZE * GRID_SIZE
HEIGHT = SIZE * GRID_SIZE

# 颜色定义
COLORS = {
    "background": (255, 255, 255),
    "grid": (200, 200, 200),
    "agent": (0, 128, 255),
    "trap": (255, 0, 0),
    "treasure": (255, 215, 0),
    "path": (173, 216, 230)
}


# 初始化
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning Maze")
clock = pygame.time.Clock()


class MAP_ENTRY_TYPE(Enum):
    MAP_EMPTY = 0,
    MAP_BLOCK = 1,


class WALL_DIRECTION(Enum):
    WALL_LEFT = 0,
    WALL_UP = 1,
    WALL_RIGHT = 2,
    WALL_DOWN = 3,


def find_set(parent: list[int], index: int):
    if index != parent[index]:
        return find_set(parent, parent[index])
    return parent[index]


# union two unconnected trees
def union_set(parent: list[int], index1: int, index2: int, weightlist: list[int]):
    root1 = find_set(parent, index1)
    root2 = find_set(parent, index2)
    if root1 == root2:
        return
    if root1 != root2:
        # take the high weight tree as the root,
        # make the whole tree balance to achieve everage search time O(logN)
        if weightlist[root1] > weightlist[root2]:
            parent[root2] = root1
            weightlist[root1] += weightlist[root2]
        else:
            parent[root1] = root2
            weightlist[root2] += weightlist[root2]


class Map:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.map = np.ones((height, width), dtype=int)
        # self.map = [[0 for _ in range(self.width)] for _ in range(self.height)]

    def reset_map(self, value: MAP_ENTRY_TYPE):
        for y in range(self.height):
            for x in range(self.width):
                self.set_map(x, y, value)

    def set_map(self, x, y, value):
        if value == MAP_ENTRY_TYPE.MAP_EMPTY:
            self.map[y, x] = 0
        elif value == MAP_ENTRY_TYPE.MAP_BLOCK:
            self.map[y, x] = 1

    def is_visited(self, x, y):
        return self.map[y, x] == 0

    def check_adjacent_pos(self, x, y, width, height, parentlist: list, weightlist: list):
        directions = []

        def _get_index(_x, _y):
            return _x * height + _y

        node1 = _get_index(x, y)
        root1 = find_set(parentlist, node1)
        # check four adjacent entries, add any unconnected entries
        if x > 0:
            root2 = find_set(parentlist, _get_index(x - 1, y))
            if root1 != root2:
                directions.append(WALL_DIRECTION.WALL_LEFT)

        if y > 0:
            root2 = find_set(parentlist, _get_index(x, y - 1))
            if root1 != root2:
                directions.append(WALL_DIRECTION.WALL_UP)

        if x < width - 1:
            root2 = find_set(parentlist, _get_index(x + 1, y))
            if root1 != root2:
                directions.append(WALL_DIRECTION.WALL_RIGHT)

        if y < height - 1:
            root2 = find_set(parentlist, _get_index(x, y + 1))
            if root1 != root2:
                directions.append(WALL_DIRECTION.WALL_DOWN)

        if len(directions):
            # choose one of the unconnected adjacent entries
            direction = choice(directions)
            if direction == WALL_DIRECTION.WALL_LEFT:
                adj_x, adj_y = (x - 1, y)
                self.set_map(2 * x, 2 * y + 1, MAP_ENTRY_TYPE.MAP_EMPTY)
            elif direction == WALL_DIRECTION.WALL_UP:
                adj_x, adj_y = (x, y - 1)
                self.set_map(2 * x + 1, 2 * y, MAP_ENTRY_TYPE.MAP_EMPTY)
            elif direction == WALL_DIRECTION.WALL_RIGHT:
                adj_x, adj_y = (x + 1, y)
                self.set_map(2 * x + 2, 2 * y + 1, MAP_ENTRY_TYPE.MAP_EMPTY)
            else:
                adj_x, adj_y = (x, y + 1)
                self.set_map(2 * x + 1, 2 * y + 2, MAP_ENTRY_TYPE.MAP_EMPTY)

            node2 = _get_index(adj_x, adj_y)
            union_set(parentlist, node1, node2, weightlist)
            return True
        else:
            # the four adjacent entries are all connected, so can remove this entry
            return False

    def union_find_set(self):
        width, height = (self.width - 1) // 2, (self.height - 1) // 2
        # find the root of the tree which the node belongs to

        parentlist = [x * height + y for x in range(width) for y in range(height)]
        weightlist = [1 for _ in range(width) for _ in range(height)]
        checklist = []
        for x in range(width):
            for y in range(height):
                checklist.append((x, y))
                # set all entries to empty
                self.set_map(2 * x + 1, 2 * y + 1, MAP_ENTRY_TYPE.MAP_EMPTY)

        while len(checklist):
            # select a random entry from checklist
            entry = choice(checklist)
            if not self.check_adjacent_pos(entry[0], entry[1], width, height, parentlist, weightlist):
                checklist.remove(entry)

        return self


class VisualGridWorld:
    def __init__(self):
        self.m = Map(SIZE, SIZE).union_find_set()
        self.goal = (SIZE - 2, SIZE - 2)
        self.state = (1, 1)
        self.path_history = []

    def reset(self):
        self.state = (1, 1)
        return self._pos_to_state(self.state)

    def _pos_to_state(self, pos):
        return pos[0] * SIZE + pos[1]

    def step(self, action):
        x, y = self.state
        if action == 0:
            x = max(x - 1, 0)
        elif action == 1:
            x = min(x + 1, SIZE - 1)
        elif action == 2:
            y = max(y - 1, 0)
        elif action == 3:
            y = min(y + 1, SIZE - 1)

        new_state = (x, y)
        reward = -1

        if self.m.map[y, x] == 1:
            reward = -50
        elif new_state == self.goal:
            reward = SIZE ** 2

        done = new_state == self.goal or self.m.map[y, x] == 1
        self.state = new_state
        return self._pos_to_state(new_state), reward, done


class VisualQLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=10.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(4)
        else:
            return np.argmax(q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        predict = q_table[state, action]
        target = reward + self.gamma * np.max(q_table[next_state])
        q_table[state, action] += self.alpha * (target - predict)


def draw_grid(surface):
    # 绘制网格线
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(surface, COLORS["grid"], (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(surface, COLORS["grid"], (0, y), (WIDTH, y))


def render(screen, env: VisualGridWorld, reward, path):
    screen.fill(COLORS["background"])

    # 绘制特殊格子
    for x in range(SIZE):
        for y in range(SIZE):
            rect = pygame.Rect(y * GRID_SIZE, x * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            if (x, y) == env.goal:
                pygame.draw.rect(screen, COLORS["treasure"], rect)
            elif env.m.map[y, x] == 1:
                pygame.draw.rect(screen, COLORS["trap"], rect)

    # 绘制历史路径
    for state in path:
        x = state // SIZE
        y = state % SIZE
        center = (y * GRID_SIZE + GRID_SIZE // 2, x * GRID_SIZE + GRID_SIZE // 2)
        pygame.draw.circle(screen, COLORS["path"], center, GRID_SIZE // 4)

    # 绘制当前代理
    x, y = env.state
    center = (y * GRID_SIZE + GRID_SIZE // 2, x * GRID_SIZE + GRID_SIZE // 2)
    pygame.draw.circle(screen, COLORS["agent"], center, GRID_SIZE // 3)

    # 显示训练信息
    font = pygame.font.Font(None, 36)
    text = font.render(f"Episode: {episode}  Reward: {reward}", True, (0, 0, 0))
    screen.blit(text, (10, 10))

    draw_grid(screen)
    pygame.display.update()
    pygame.time.delay(10)


env = VisualGridWorld()
agent = VisualQLearningAgent()
q_table = np.zeros((SIZE**2, 4))
clock.tick(60)  # 限制60FPS
# 主循环
running = True
episode = 0

while running:
    # 执行一个训练episode
    state = env.reset()
    total_reward = 0
    done = False
    path = [SIZE + 1]

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        path.append(state)
        if episode % 100 == 0:
            render(screen, env, total_reward, path)

    agent.epsilon *= 0.995
    episode += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False


pygame.quit()
