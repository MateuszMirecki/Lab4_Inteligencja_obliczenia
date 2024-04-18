import numpy as np
import gymnasium as gym
from gymnasium import spaces

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

    def _get_obs(self):
        return {"agent": self._agent_location, "items": self._item_locations}

    def reset(self, seed=None, options=None):
        self._agent_location = np.random.randint(0, self.size, size=2)
        self._item_locations = np.array([np.random.randint(0, self.size, size=2) for _ in range(self.num_items)])
        return self._get_obs()

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        reward = 0
        terminated = False
        
        # Check for item collection
        new_items = []
        for item in self._item_locations:
            if np.array_equal(item, self._agent_location):
                reward += 1  # Collect item
            else:
                new_items.append(item)
        self._item_locations = np.array(new_items)
        
        # Check if all items are collected
        if len(self._item_locations) == 0:
            terminated = True
        
        return self._get_obs(), reward, terminated, {}

    def render(self):
        grid = np.full((self.size, self.size), ' ')
        x, y = self._agent_location
        grid[y, x] = 'A'  # Agent
        for item in self._item_locations:
            ix, iy = item
            grid[iy, ix] = 'I'  # Item
        print("\n".join(''.join(row) for row in grid))
        print()

import pygame
import numpy as np
from pygame.locals import QUIT

class PygameRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        
    def render(self, agent_pos, item_positions):
        self.screen.fill((0, 0, 0))  # Clear the screen
        
        # Draw agent (blue square)
        pygame.draw.rect(self.screen, (0, 0, 255), (agent_pos[0] * 30, agent_pos[1] * 30, 30, 30))
        
        # Draw items (green squares)
        for item_pos in item_positions:
            pygame.draw.rect(self.screen, (0, 255, 0), (item_pos[0] * 30, item_pos[1] * 30, 30, 30))
        
        pygame.display.flip()
        self.clock.tick(4)  # Cap the frame rate at 4 FPS

class ItemCollectorEnvWithPygameRenderer(ItemCollectorEnv):
    def __init__(self, size=5, num_items=3):
        super().__init__(size, num_items)
        self.renderer = PygameRenderer(size * 30, size * 30)
    
    def render(self):
        self.renderer.render(self._agent_location, self._item_locations)

def automatic_play_2D(size=5, num_items=3):
    env = ItemCollectorEnvWithPygameRenderer(size, num_items)
    obs = env.reset()
    done = False
    step = 0
    total_reward = 0
    while not done:
        for event in pygame.event.get():
            if event.type == QUIT:
                return
        
        # Calculate closest item
        agent_pos = obs['agent']
        closest_item = None
        closest_distance = float('inf')
        for item in obs['items']:
            distance = np.abs(item - agent_pos).sum()
            if distance < closest_distance:
                closest_distance = distance
                closest_item = item
        
        # Decide action based on closest item
        if closest_item is not None:
            direction = closest_item - agent_pos
            action = np.argmax(np.abs(direction))
            if direction[action] < 0:
                action += 2  # To map from indices [0, 1] to actions [2, 3]
        else:
            # If no items left, just take a random action
            action = env.action_space.sample()
        
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1
        
        # Render the current state
        env.render()  
        # Delay to control the speed of simulation
        pygame.time.wait(100)  # Wait for 0.1 second
        
        if done:
            print(f"Solved in {step} steps with total reward: {total_reward}")
            env.render()  
            pygame.time.wait(1000) 
            pygame.quit()

