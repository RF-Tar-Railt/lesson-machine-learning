import sys
import numpy as np
from typing import Optional
from enum import Enum, auto
from random import Random
import pygame

# 游戏参数
SIZE_MAX = 45
GRID_SIZE = 720 // SIZE_MAX
WIDTH = SIZE_MAX * GRID_SIZE
HEIGHT = SIZE_MAX * GRID_SIZE

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
clock = pygame.time.Clock()


class MAP_ENTRY_TYPE(Enum):
    """地图单元类型枚举"""
    MAP_EMPTY = auto()
    MAP_BLOCK = auto()


class WALL_DIRECTION(Enum):
    """墙壁方向枚举"""
    WALL_LEFT = auto()
    WALL_UP = auto()
    WALL_RIGHT = auto()
    WALL_DOWN = auto()


def find_set(parent: list[int], index: int) -> int:
    """查找并返回元素所属集合的根节点，使用路径压缩优化"""
    if index != parent[index]:
        parent[index] = find_set(parent, parent[index])  # 路径压缩
    return parent[index]


def union_set(parent: list[int], index1: int, index2: int, weight_list: list[int]) -> None:
    """合并两个不相连的树，使用按秩合并优化"""
    root1 = find_set(parent, index1)
    root2 = find_set(parent, index2)

    if root1 == root2:
        return

    # 按秩合并：将较小的树连接到较大的树上
    if weight_list[root1] > weight_list[root2]:
        parent[root2] = root1
        weight_list[root1] += weight_list[root2]
    else:
        parent[root1] = root2
        weight_list[root2] += weight_list[root1]  # 修复了这里的错误，原来是加了两次root2


class Maze:
    """迷宫生成类"""

    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        """
        初始化迷宫

        Args:
            width: 迷宫宽度
            height: 迷宫高度
            seed: 随机数种子
        """
        self.width = width
        self.height = height
        self.map = np.ones((height, width), dtype=int)
        self.rd = Random(seed)  # 直接传入种子

    def reset_map(self, value: MAP_ENTRY_TYPE) -> None:
        """重置地图所有单元为指定值"""
        fill_value = 0 if value == MAP_ENTRY_TYPE.MAP_EMPTY else 1
        self.map.fill(fill_value)  # 使用numpy的fill方法，更高效

    def set_map(self, x: int, y: int, value: MAP_ENTRY_TYPE) -> None:
        """设置地图特定位置的值"""
        if 0 <= x < self.width and 0 <= y < self.height:  # 添加边界检查
            self.map[y, x] = 0 if value == MAP_ENTRY_TYPE.MAP_EMPTY else 1

    def is_visited(self, x: int, y: int) -> bool:
        """检查指定位置是否已访问"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.map[y, x] == 0
        return False

    def check_adjacent_pos(self, x: int, y: int, width: int, height: int,
                           parent_list: list[int], weight_list: list[int]) -> bool:
        """检查并处理相邻位置"""
        directions = []

        def _get_index(_x: int, _y: int) -> int:
            return _x * height + _y

        node1 = _get_index(x, y)
        root1 = find_set(parent_list, node1)

        # 检查四个相邻位置，添加任何未连接的位置
        adjacent_positions = [
            (x - 1, y, WALL_DIRECTION.WALL_LEFT, 2 * x, 2 * y + 1),
            (x, y - 1, WALL_DIRECTION.WALL_UP, 2 * x + 1, 2 * y),
            (x + 1, y, WALL_DIRECTION.WALL_RIGHT, 2 * x + 2, 2 * y + 1),
            (x, y + 1, WALL_DIRECTION.WALL_DOWN, 2 * x + 1, 2 * y + 2)
        ]

        for adj_x, adj_y, direction, wall_x, wall_y in adjacent_positions:
            if 0 <= adj_x < width and 0 <= adj_y < height:  # 边界检查
                root2 = find_set(parent_list, _get_index(adj_x, adj_y))
                if root1 != root2:
                    directions.append((direction, adj_x, adj_y, wall_x, wall_y))

        if directions:
            # 随机选择一个未连接的相邻位置
            direction, adj_x, adj_y, wall_x, wall_y = self.rd.choice(directions)

            # 打通墙壁
            self.set_map(wall_x, wall_y, MAP_ENTRY_TYPE.MAP_EMPTY)

            # 合并两个集合
            node2 = _get_index(adj_x, adj_y)
            union_set(parent_list, node1, node2, weight_list)
            return True
        else:
            # 四个相邻位置都已连接，可以移除此位置
            return False

    def union_find_set(self) -> 'Maze':
        """使用并查集算法生成迷宫"""
        width, height = (self.width - 1) // 2, (self.height - 1) // 2

        # 初始化并查集数据结构
        parent_list = list(range(width * height))
        weight_list = [1] * (width * height)

        # 初始化检查列表和迷宫
        check_list = [(x, y) for x in range(width) for y in range(height)]

        # 设置所有单元为空
        for x in range(width):
            for y in range(height):
                self.set_map(2 * x + 1, 2 * y + 1, MAP_ENTRY_TYPE.MAP_EMPTY)

        # 迷宫生成主循环
        while check_list:
            # 随机选择一个位置
            idx = self.rd.randrange(len(check_list))
            entry = check_list[idx]

            if not self.check_adjacent_pos(entry[0], entry[1], width, height, parent_list, weight_list):
                check_list.pop(idx)  # 使用pop(idx)而不是remove，更高效

        return self


# 初始配置界面类
class ConfigWindow:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        self.font = pygame.font.Font(None, 32)
        self.input_box = pygame.Rect(100, 100, 200, 32)
        self.color_inactive = pygame.Color('lightskyblue3')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.text = ''
        self.active = False
        self.done = False

    def run(self):
        pygame.display.set_caption("Maze Configuration")
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.input_box.collidepoint(event.pos):
                        self.active = not self.active
                    else:
                        self.active = False
                    self.color = self.color_active if self.active else self.color_inactive
                if event.type == pygame.KEYDOWN and self.active:
                    if event.key == pygame.K_RETURN:
                        self.done = True
                    elif event.key == pygame.K_BACKSPACE:
                        self.text = self.text[:-1]
                    elif event.unicode.isdigit():
                        self.text += event.unicode

            self.screen.fill((255, 255, 255))
            txt_surface = self.font.render("Enter maze size (15-45): ", True, self.color)
            self.screen.blit(txt_surface, (self.input_box.x - 40, self.input_box.y - 40))
            input_surface = self.font.render(self.text, True, self.color)
            self.screen.blit(input_surface, (self.input_box.x + 5, self.input_box.y + 5))
            pygame.draw.rect(self.screen, self.color, self.input_box, 2)
            pygame.display.flip()

        size = max(15, min(SIZE_MAX, int(self.text) if self.text else 15))

        self.done = False
        self.text = ''
        self.active = False
        self.color = self.color_inactive

        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.input_box.collidepoint(event.pos):
                        self.active = not self.active
                    else:
                        self.active = False
                    self.color = self.color_active if self.active else self.color_inactive
                if event.type == pygame.KEYDOWN and self.active:
                    if event.key == pygame.K_RETURN:
                        self.done = True
                    elif event.key == pygame.K_BACKSPACE:
                        self.text = self.text[:-1]
                    elif event.unicode.isdigit():
                        self.text += event.unicode

            self.screen.fill((255, 255, 255))
            txt_surface = self.font.render("Enter Episode: ", True, self.color)
            self.screen.blit(txt_surface, (self.input_box.x - 40, self.input_box.y - 40))
            input_surface = self.font.render(self.text, True, self.color)
            self.screen.blit(input_surface, (self.input_box.x + 5, self.input_box.y + 5))
            pygame.draw.rect(self.screen, self.color, self.input_box, 2)
            pygame.display.flip()

        episode = max(1000, int(self.text) if self.text else 1000)
        return size, episode


class MazeEnv:
    def __init__(self, size=5):
        self.size = size
        self.maze = Maze(size, size, 233).union_find_set().map
        self.start_pos = (1, 1)
        self.goal_pos = (self.size - 2, self.size - 2)
        self.current_pos = self.start_pos

    def reset(self) -> tuple[int, int]:
        self.current_pos = self.start_pos
        return self.current_pos

    def get_valid_actions(self):
        x, y = self.current_pos
        actions = []
        if self.maze[x - 1, y] == 0:
            actions.append(0)
        if x < self.size - 1 and self.maze[x + 1, y] == 0:
            actions.append(1)
        if self.maze[x, y - 1] == 0:
            actions.append(2)
        if y < self.size - 1 and self.maze[x, y + 1] == 0:
            actions.append(3)
        return actions

    def step(self, action) -> tuple[tuple[int, int], int, bool]:
        x, y = self.current_pos

        if action == 0:
            new_pos = (x - 1, y)
        elif action == 1:
            new_pos = (x + 1, y)
        elif action == 2:
            new_pos = (x, y - 1)
        else:
            new_pos = (x, y + 1)

        # 边界检查
        if new_pos[0] < 1 or new_pos[0] >= self.size or new_pos[1] < 1 or new_pos[1] >= self.size:
            return self.current_pos, -100, True

        # 障碍物检查
        if self.maze[x, y] == 1:
            return self.current_pos, -50, True

        self.current_pos = new_pos

        if new_pos == self.goal_pos:
            return new_pos, self.size ** 2, True

        return new_pos, -1, False


class QLearningAgent:
    def __init__(self, size):
        self.q_table = np.zeros((size, size, 4))  # 状态空间，4 个动作
        self.alpha = 0.1  # 学习率
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 0.5  # epsilon-greedy 策略
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def choose_action(self, state, actions):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(actions)
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state[0], state[1], action]
        max_next_q = np.max(self.q_table[next_state[0], next_state[1]])
        target = reward + self.gamma * max_next_q
        self.q_table[state[0], state[1], action] += self.alpha * (target - current_q)


class MazeGUI:
    def __init__(self, env: MazeEnv):
        pygame.init()
        self.size = env.maze.shape[0]
        cell_size = 720 // self.size
        self.screen = pygame.display.set_mode((self.size * cell_size, self.size * cell_size))
        self.cell_size = cell_size
        self.clock = pygame.time.Clock()
        self.env = env

    def draw_maze(self, path):
        self.screen.fill((255, 255, 255))

        for x in range(self.size):
            for y in range(self.size):
                rect = (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)

                if self.env.maze[x, y] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
                elif (x, y) == self.env.goal_pos:
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        for state in path[:-1]:
            x, y = state
            pygame.draw.circle(
                self.screen,
                (0, 150, 0),
                (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2),
                self.cell_size // 4,
            )

        pygame.draw.circle(
            self.screen,
            (0, 255, 0),
            (path[-1][0] * self.cell_size + self.cell_size // 2, path[-1][1] * self.cell_size + self.cell_size // 2),
            self.cell_size // 3,
        )

        pygame.display.flip()


def train(gui: MazeGUI, episodes=20000):
    agent = QLearningAgent(gui.size)

    for episode in range(episodes):
        state = gui.env.reset()
        total_reward = 0
        done = False
        path = [state]
        pygame.display.set_caption(f"Training, Episode: {episode}")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            action = agent.choose_action(state, gui.env.get_valid_actions())
            next_state, reward, done = gui.env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            path.append(state)
            total_reward += reward
            if episode % 100 == 0:
                gui.draw_maze(path)
                gui.clock.tick(60)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if episode % 100 == 0:
            print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

    return agent


def run(gui: MazeGUI, agent: QLearningAgent):
    state = gui.env.reset()
    done = False
    pygame.display.set_caption("Running")
    path = [state]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not done:
            action = agent.choose_action(state, gui.env.get_valid_actions())
            next_state, reward, done = gui.env.step(action)
            state = next_state
            path.append(state)

        gui.draw_maze(path)
        gui.clock.tick(5)


if __name__ == "__main__":
    # 获取迷宫大小
    config = ConfigWindow()
    size, episodes = config.run()
    if size % 2 == 0:
        size += 1
    pygame.quit()

    # 创建环境并训练
    gui = MazeGUI(MazeEnv(size=size))
    agent = train(gui, episodes=episodes)
    # 运行
    run(gui, agent)
