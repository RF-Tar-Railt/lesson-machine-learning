import sys
import numpy as np
from enum import Enum
from random import Random
import pygame

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
        self.rd = Random()
        self.rd.seed(123)
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
            direction = self.rd.choice(directions)
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
            entry = self.rd.choice(checklist)
            if not self.check_adjacent_pos(entry[0], entry[1], width, height, parentlist, weightlist):
                checklist.remove(entry)

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
                if event.type == pygame.KEYDOWN:
                    if self.active:
                        if event.key == pygame.K_RETURN:
                            self.done = True
                        elif event.key == pygame.K_BACKSPACE:
                            self.text = self.text[:-1]
                        else:
                            if event.unicode.isdigit():
                                self.text += event.unicode

            self.screen.fill((255, 255, 255))
            txt_surface = self.font.render("Enter maze size (15-45): ", True, self.color)
            self.screen.blit(txt_surface, (self.input_box.x - 40, self.input_box.y - 40))
            input_surface = self.font.render(self.text, True, self.color)
            self.screen.blit(input_surface, (self.input_box.x + 5, self.input_box.y + 5))
            pygame.draw.rect(self.screen, self.color, self.input_box, 2)
            pygame.display.flip()

        return max(15, min(45, int(self.text) if self.text else 15))


class MazeEnv:
    def __init__(self, size=5):
        self.SIZE = size
        self.maze = Map(SIZE, SIZE).union_find_set().map
        self.start_pos = (1, 1)
        self.goal_pos = (self.SIZE - 2, self.SIZE - 2)
        self.current_pos = self.start_pos

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos

    def get_valid_actions(self):
        x, y = self.current_pos
        actions = []
        if self.maze[x - 1, y] == 0:
            actions.append(0)
        if x < self.SIZE - 1 and self.maze[x + 1, y] == 0:
            actions.append(1)
        if self.maze[x, y - 1] == 0:
            actions.append(2)
        if y < self.SIZE - 1 and self.maze[x, y + 1] == 0:
            actions.append(3)
        return actions

    def step(self, action):
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
        if new_pos[0] < 1 or new_pos[0] >= self.SIZE or new_pos[1] < 1 or new_pos[1] >= self.SIZE:
            return self.current_pos, -100, True

        # 障碍物检查
        if self.maze[x, y] == 1:
            return self.current_pos, -50, True

        self.current_pos = new_pos

        if new_pos == self.goal_pos:
            return new_pos, self.SIZE ** 2, True

        return new_pos, -1, False


class QLearningAgent:
    def __init__(self, size):
        self.q_table = np.zeros((size, size, 4))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.5
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.0001

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
        self.SIZE = env.maze.shape[0]
        cell_size = 720 // self.SIZE
        self.size = (self.SIZE * cell_size, self.SIZE * cell_size)
        self.screen = pygame.display.set_mode(self.size)
        self.cell_size = cell_size
        self.clock = pygame.time.Clock()
        self.env = env

    def draw_maze(self, path):
        self.screen.fill((255, 255, 255))

        for x in range(self.SIZE):
            for y in range(self.SIZE):
                rect = (x * self.cell_size, y * self.cell_size,
                        self.cell_size, self.cell_size)

                if self.env.maze[x, y] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
                elif (x, y) == self.env.goal_pos:
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        for state in path[:-1]:
            x, y = state
            pygame.draw.circle(self.screen, (0, 150, 0),
                               (x * self.cell_size + self.cell_size // 2,
                                y * self.cell_size + self.cell_size // 2),
                               self.cell_size // 4)

        pygame.draw.circle(self.screen, (0, 255, 0),
                           (path[-1][0] * self.cell_size + self.cell_size // 2,
                            path[-1][1] * self.cell_size + self.cell_size // 2),
                           self.cell_size // 3)

        pygame.display.flip()

    def run(self, env, episodes=20000):

        agent = QLearningAgent(self.SIZE)

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            path = [env.start_pos]
            pygame.display.set_caption(f"Training, Episode: {episode}")

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                action = agent.choose_action(state, env.get_valid_actions())
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state)
                state = next_state
                path.append(state)
                total_reward += reward
                if episode % 100 == 0:
                    self.draw_maze(path)
                    self.clock.tick(60)

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            if episode % 100 == 0:
                print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

        state = env.reset()
        done = False
        pygame.display.set_caption("Testing")
        path = [env.start_pos]

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if not done:
                action = agent.choose_action(state, env.get_valid_actions())
                next_state, reward, done = env.step(action)
                state = next_state
                path.append(state)

            self.draw_maze(path)
            self.clock.tick(5)


if __name__ == "__main__":
    # 获取迷宫大小
    config = ConfigWindow()
    SIZE = config.run()
    if SIZE % 2 == 0:
        SIZE += 1
    pygame.quit()

    # 创建环境并训练
    env = MazeEnv(size=SIZE)
    # 运行可视化
    gui = MazeGUI(env)
    gui.run(env)
