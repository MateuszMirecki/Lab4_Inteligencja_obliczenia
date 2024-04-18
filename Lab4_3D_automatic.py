import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ItemCollectorEnv3D(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, size=5, num_items=3):
        super().__init__()
        self.size = size
        self.num_items = num_items
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=size - 1, shape=(3,), dtype=int),
            "items": spaces.Box(low=0, high=size - 1, shape=(num_items, 3), dtype=int),
        })
        self.action_space = spaces.Discrete(6)  # Adding up and down movements in the z-axis
        self._action_to_direction = {0: np.array([1, 0, 0]), 1: np.array([0, 1, 0]), 
                                     2: np.array([-1, 0, 0]), 3: np.array([0, -1, 0]),
                                     4: np.array([0, 0, 1]), 5: np.array([0, 0, -1])}  # z-axis control

    def _get_obs(self):
        return {"agent": self._agent_location, "items": self._item_locations}

    def reset(self, seed=None, options=None):
        self._agent_location = np.random.randint(0, self.size, size=3)
        self._item_locations = np.array([np.random.randint(0, self.size, size=3) for _ in range(self.num_items)])
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
        print(f"Agent Position: {self._agent_location}")
        print("Items remaining:")
        for item in self._item_locations:
            print(item)
        print()

def automatic_play_3D():
    env = ItemCollectorEnv3D(size=6, num_items=6)
    obs = env.reset()
    done = False
    total_reward = 0
    env.render()
    steps = 0

    while not done:
        action = np.random.randint(0, 6)  # Choose a random action
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        steps += 1
        if done:
            print(f"Game completed in {steps} steps with total reward: {total_reward}")
            input("Simulation ended , click enter to close.")


