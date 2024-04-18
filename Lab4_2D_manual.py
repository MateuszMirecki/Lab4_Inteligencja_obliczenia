import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from pygame.locals import KEYDOWN, K_UP, K_DOWN, K_LEFT, K_RIGHT, QUIT

class ItemCollectorEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, size=5, num_items=3):
        super().__init__()
        self.size = size
        self.num_items = num_items
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "items": spaces.Box(0, size - 1, shape=(num_items, 2), dtype=int),
        })
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {0: np.array([1, 0]), 1: np.array([0, 1]), 
                                     2: np.array([-1, 0]), 3: np.array([0, -1])}
        
        # Initialize Pygame
        pygame.init()
        self.cell_size = 50
        self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

    def _get_obs(self):
        return {"agent": self._agent_location, "items": self._item_locations}

    def reset(self):
        self._agent_location = np.random.randint(0, self.size, size=2)
        self._item_locations = np.array([np.random.randint(0, self.size, size=2) for _ in range(self.num_items)])
        return self._get_obs()

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        reward = 0
        terminated = False
        
        new_items = []
        for item in self._item_locations:
            if np.array_equal(item, self._agent_location):
                reward += 1  # Collect item
            else:
                new_items.append(item)
        self._item_locations = np.array(new_items)
        
        if len(self._item_locations) == 0:
            terminated = True
        
        return self._get_obs(), reward, terminated, {}

    def render(self):
        self.screen.fill((0, 0, 0))  # Black background
        for item in self._item_locations:
            pygame.draw.rect(self.screen, (0, 255, 0), (item[0] * self.cell_size, item[1] * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (0, 0, 255), (self._agent_location[0] * self.cell_size, self._agent_location[1] * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        pygame.quit()

def manual_play_2D(size=5, num_items=3):
    env = ItemCollectorEnv(size, num_items)
    obs = env.reset()
    done = False
    total_reward = 0
    env.render()

    running = True
    while not done and running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_UP:
                    action = 3
                elif event.key == K_DOWN:
                    action = 1
                elif event.key == K_LEFT:
                    action = 2
                elif event.key == K_RIGHT:
                    action = 0
                else:
                    continue
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                env.render()
        if done:
            print(f"Game completed with total reward: {total_reward}")
            pygame.time.wait(2000)  # Delay to see final state
            running = False

    env.close()

if __name__ == "__main__":
    manual_play_2D()
