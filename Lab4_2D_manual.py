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
        self.actions = {'d': 0, 's': 1, 'a': 2, 'w': 3}  

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
        grid = np.full((self.size, self.size), ' ')
        x, y = self._agent_location
        grid[y, x] = 'A'  # Agent
        for item in self._item_locations:
            ix, iy = item
            grid[iy, ix] = 'I'  # Item
        print("\n".join(''.join(row) for row in grid))
        print()

def manual_play_2D(size=5, num_items=3):
    env = ItemCollectorEnv(size, num_items)
    obs = env.reset()
    done = False
    total_reward = 0
    env.render()

    while not done:
        move = input("Enter your move (w=up, s=down, a=left, d=right): ")
        if move in env.actions:
            obs, reward, done, _ = env.step(env.actions[move])
            total_reward += reward
            env.render()
            if done:
                print(f"Game completed in with total reward: {total_reward}")
                input("Thanks for completing the game, click enter to close")
        else:
            print("Invalid move! Use 'w', 's', 'a', or 'd'.")


