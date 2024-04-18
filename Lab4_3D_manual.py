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
        self.actions = {'d': 0, 's': 1, 'a': 2, 'w': 3, 'e': 4, 'q': 5}  # Adding new keys for z-axis

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
        # Visualization for 3D is tricky in a 2D terminal, might be better suited for GUI-based rendering.
        print(f"Agent Position: {self._agent_location}")
        print("Items remaining:")
        for item in self._item_locations:
            print(item)
        print()

def manual_play_3D():
    env = ItemCollectorEnv3D(size=6, num_items=6)
    obs = env.reset()
    done = False
    total_reward = 0
    env.render()
    print(" ===========================================\n\n 2c + 1 means second coordinate will increase by one \n\n===========================================\n")
    while not done:
        move = input("Enter your move \n d=right (1c + 1) || a=left (1c - 1)  \n  w=up (2c - 1 ) || s=down (2c + 1) \n e=ascend (3c + 1) ||  q=descend(3c - 1): ")
        if move in env.actions:
            obs, reward, done, _ = env.step(env.actions[move])
            total_reward += reward
            env.render()
            if done:
                print(f"Game completed with total reward: {total_reward}")
                input("Thanks for completing the game, click enter to close")
        else:
            print("Invalid move! Use 'w', 's', 'a', 'd', 'e', or 'q'.")

