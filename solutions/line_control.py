import gym
import numpy as np
import pygame
from pygame import gfxdraw
from math import sqrt, exp, fabs


HORIZON = 500
ACCELERATION_MIN = 500
PENALTY = -1000.


class SimpleLineControlGymEnv(gym.Env):
    """This class mimics an OpenAI Gym environment"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, env_config=None):
        """Initialize GymDomain.
        # Parameters
        gym_env: The Gym environment (gym.env) to wrap.
        """
        inf = np.finfo(np.float32).max
        self.action_space = gym.spaces.Box(
            np.array([-1000.0, -1000.0]), np.array([1000.0, 1000.0]), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            np.array([-inf, -inf, -inf, -inf]),
            np.array([inf, inf, inf, inf]),
            dtype=np.float32,
        )
        self._delta_t = 0.001
        self._init_pos_x = 0.0
        self._init_pos_y = 0.5
        self._init_speed_x = 10.0
        self._init_speed_y = 1.0
        self._pos_x = None
        self._pos_y = None
        self._speed_x = None
        self._speed_y = None
        self._path = []
        
        self.screen = None
        self.clock = None
        self.isopen = True

    def get_state(self):
        return np.array(
            [self._pos_x, self._pos_y, self._speed_x, self._speed_y], dtype=np.float32
        )

    def set_state(self, state):
        self._pos_x = state[0]
        self._pos_y = state[1]
        self._speed_x = state[2]
        self._speed_y = state[3]

    def reset(self):
        self._pos_x = self._init_pos_x
        self._pos_y = self._init_pos_y
        self._speed_x = self._init_speed_x
        self._speed_y = self._init_speed_y
        self._path = []
        return np.array(
            [self._pos_x, self._pos_y, self._speed_x, self._speed_y], dtype=np.float32
        )

    def step(self, action):
        if sqrt(action[0]*action[0] + action[1]*action[1]) < ACCELERATION_MIN:
            obs = np.array(
                [self._pos_x, self._pos_y, self._speed_x, self._speed_y], dtype=np.float32
            )
            return obs, PENALTY, True, {}
        self._speed_x = self._speed_x + action[0] * self._delta_t
        self._speed_y = self._speed_y + action[1] * self._delta_t
        if self._speed_x < 0.:
            obs = np.array(
                [self._pos_x, self._pos_y, self._speed_x, self._speed_y], dtype=np.float32
            )
            return obs, PENALTY, True, {}
        self._pos_x = self._pos_x + self._delta_t * self._speed_x
        self._pos_y = self._pos_y + self._delta_t * self._speed_y
        obs = np.array(
            [self._pos_x, self._pos_y, self._speed_x, self._speed_y], dtype=np.float32
        )
        reward = exp(-fabs(self._pos_y))
        done = bool(fabs(self._pos_y) > 1.0)
        self._path.append((self._pos_x, self._pos_y))
        return obs, reward if not done else PENALTY, done, {}

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))
        self.track = gfxdraw.hline(
            self.surf,
            0,
            screen_width,
            int(screen_height / 2),
            (0, 0, 255)
        )

        if len(self._path) > 1:
            for p in range(len(self._path) - 1):
                gfxdraw.line(
                    self.surf,
                    int(self._path[p][0] * 100),
                    int(screen_height / 2 + self._path[p][1] * 100),
                    int(self._path[p+1][0] * 100),
                    int(screen_height / 2 + self._path[p+1][1] * 100),
                    (255, 0, 0)
                )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
