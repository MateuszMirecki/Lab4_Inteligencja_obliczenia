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

# Agent solving the problem with random actions
def solve_item_collector():
    env = ItemCollectorEnv(size=5, num_items=3)
    obs = env.reset()
    done = False
    step = 0
    total_reward = 0
    while not done:
        env.render()  # Print the grid to the terminal
        action = env.action_space.sample()  # Random action
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1
        if done:
            print(f"Solved in {step} steps with total reward: {total_reward}")
            env.render()  # Final state

solve_item_collector()
