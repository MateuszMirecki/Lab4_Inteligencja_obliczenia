import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {0: np.array([1, 0]), 1: np.array([0, 1]), 
                                     2: np.array([-1, 0]), 3: np.array([0, -1])}
        
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def reset(self, seed=None, options=None):
        self._agent_location = np.random.randint(0, self.size, size=2)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.random.randint(0, self.size, size=2)
        return self._get_obs()

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0
        return self._get_obs(), reward, terminated, {}

    def render(self):
        grid = np.full((self.size, self.size), ' ')
        x, y = self._agent_location
        tx, ty = self._target_location
        grid[ty, tx] = 'T'  # Target
        grid[y, x] = 'A'  # Agent
        print("\n".join(''.join(row) for row in grid))
        print()

# Agent solving the problem with random actions
def solve_grid_world():
    env = GridWorldEnv(size=5)
    obs = env.reset()
    done = False
    step = 0
    while not done:
        env.render()  # Print the grid to the terminal
        action = env.action_space.sample()  # Random action
        obs, reward, done, _ = env.step(action)
        step += 1
        if done:
            print(f"Solved in {step} steps")
            env.render()  # Final state

solve_grid_world()
