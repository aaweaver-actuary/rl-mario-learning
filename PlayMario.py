import cv2

import warnings
from gym.utils import passive_env_checker

import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
from collections import deque
import random, time, datetime, os, copy
import numpy as np

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

# imageio is used to save the training progress as a gif
import imageio

# pyvirtualdisplay is used to create a virtual display to watch the agent play
# from pyvirtualdisplay import Display

# progress bars:
from tqdm import tqdm

from Mario import Mario
from MetricLogger import MetricLogger
from GreyScaleObservation import GrayScaleObservation
from ResizeObservation import ResizeObservation
from SkipFrame import SkipFrame

class PlayMario:
    def __init__(self
                , world : int = 1
                , stage : int = 1
                , version : int = 3
                , render_mode : str = "rgb_array"
                , apply_api_compatibility : bool = True
                , joypad_actions : list = None
                , skip_frames : int = 4
                , resize_shape : int = 84
                , stack_frames : int = 4
                , episodes : int = 10
                , fps : int = 30
                , save_dir : str = "models"
                ):
        self.world = world
        self.stage = stage
        self.version = version
        self.render_mode = render_mode
        self.apply_api_compatibility = apply_api_compatibility
        self.skip_frames = skip_frames
        self.resize_shape = resize_shape
        self.stack_frames = stack_frames
        self.episodes = episodes
        self.fps = fps
        self.save_dir = save_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.smb_version = f"SuperMarioBros-{self.world}-{self.stage}-v{self.version}"

        if joypad_actions is None:
            self.joypad_actions = [
                # 0. walk right
                ["right"]
                
                # 1. jump right
                , ["right", "A"]
                ]
        else:
            self.joypad_actions = joypad_actions

        # initialize environment
        self.build_env()

        # initialize agent 
        self.mario = Mario(state_dim=(self.skip_frames
                                      , self.resize_shape
                                      , self.resize_shape)
                                      , action_dim=self.env.action_space.n
                                      , save_dir=self.save_dir)

        # initialize metric logger
        self.logger = MetricLogger(self.save_dir)

        # initialize replay buffer
        self.frames = []

        # calculate opencv waitkey delay
        self.opencv_wait_time = int((1000 / self.fps))

    def build_env(self):
        
        warnings.filterwarnings("ignore", category=UserWarning, module="gym.envs.registration")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=passive_env_checker.__name__)
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

      

        # initialize env based on gym version
        if gym.__version__ < "0.26":
            self.env = gym_super_mario_bros.make(self.smb_version
                                            , new_step_api=True)
        else:
            self.env = gym_super_mario_bros.make(self.smb_version
                                                 , render_mode=self.render_mode
                                                 , apply_api_compatibility=self.apply_api_compatibility
                                                 )

        # wrap env with JoypadSpace to simplify action space
        self.env = JoypadSpace(self.env, self.joypad_actions)

        # add metadata to env to allow for rendering
        self.env.metadata['render_modes'] = ['human', 'rgb_array']
        self.env.metadata['video.frames_per_second'] = self.fps
        self.env.metadata['render_fps'] = self.fps

        # reset env to start a new episode
        self.env.reset()

        # render env to show initial state
        next_state, reward, done, _, _ = self.env.step(action=0)

        # Apply Wrappers to environment
        self.env = SkipFrame(self.env, skip=self.skip_frames)
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, shape=self.resize_shape)
        if gym.__version__ < '0.26':
            self.env = FrameStack(self.env, num_stack=self.stack_frames, new_step_api=True)
        else:
            self.env = FrameStack(self.env, num_stack=self.stack_frames)

        # reset env to start a new episode
        self.env.reset()

        # Render the environment and save the frame
        # self.frame = self.env.render()

    def single_play(self):
        """
        Runs a single episode of the game
        """
        # Create a directory to store the frames for the episode
        self.save_dir = os.path.join("checkpoints", datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True)

        # Reset the environment and get the initial state of the agent
        state = self.env.reset()

        # Initialize a list to store the frames for each episode
        episode_frames = []

        # Run the episode until the game is over
        while True:
            # Get the next action from Mario based on the current state
            action = self.mario.act(state)

            # Take the action and get the next state, reward, and whether the game
            # is done
            next_state, reward, done, _, _ = self.env.step(action)

            # Render the environment and save the frame
            frame = self.env.render()
            episode_frames.append(frame)

            # Display the frame using OpenCV
            cv2.imshow('Mario Gameplay', frame)
            cv2.waitKey(self.opencv_wait_time)

            # Store the transition in the replay buffer for experience replay
            self.mario.cache(state, next_state, action, reward, done)

            # Learn from the experience replay buffer
            q, loss = self.mario.learn()

            # Log the metrics
            self.logger.log_step(reward, loss, q)
            
            # Update the state
            state = next_state

            # If the game is over, break out of the loop
            if done:
                break

        # increment episode counter
        self.episodes += 1

        # Add the episode frames to the main frames list
        self.frames.append(episode_frames)

        # Save the episode frames to a video
        print("Saving video")
        imageio.mimsave(f'games/mario_gameplay_{self.episodes}.mp4'
                        , episode_frames
                        , fps=self.fps)

    def train(self):
        """
        Runs the training loop for the agent
        """
        # Create a virtual display if on a linux server
        if os.name == "posix":
            # pyvirtualdisplay is used to create a virtual display to watch the agent play
            from pyvirtualdisplay import Display
            display = Display(visible=0, size=(1400, 900))
            display.start()

        # Initialize episode counter
        self.episodes = 0

        # loop through the episodes
        for e in tqdm(range(self.episodes), desc="Episodes"):
            print("==========================================")
            print("Episode {e} - Starting Episode")
            print("==========================================")

            # Run a single episode of the game
            self.single_play()

            # Log the episode
            self.logger.log_episode()

            # Save the model
            if e % 20 == 0:
                self.logger.record(episode=e
                                   , epsilon=self.mario.exploration_rate
                                   , step=self.mario.curr_step)

 