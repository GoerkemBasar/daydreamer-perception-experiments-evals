# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import custom_dmc
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim if present
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        
        # FIX: Output name is 'image', but input key was 'pixels'
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [pixels_shape[:2], [pixels_shape[2] * num_frames]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='image')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        # Concatenate along the last dimension (Channels) to keep (H, W, C)
        obs = np.concatenate(list(self._frames), axis=2)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ObservationDictWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        # The env now outputs a flat array named 'image', we wrap it in a dict
        self._obs_spec = {'image': env.observation_spec()}

    def reset(self):
        time_step = self._env.reset()
        return time_step._replace(observation={'image': time_step.observation})

    def step(self, action):
        time_step = self._env.step(action)
        return time_step._replace(observation={'image': time_step.observation})

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()
    
    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed):
    # 1. CUSTOM: Intercept your Walker experiment
    if name.startswith('custom_walker'):
        view_mode = name.split('_')[-1] 

        env = custom_dmc.make_custom_walker(view=view_mode, seed=seed)
        env.domain_name = 'walker'
        env.task_name = view_mode

        env = ActionRepeatWrapper(env, action_repeat)
        env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
        
        # Select camera ID based on the task name
        cam_id = 'custom_side' if view_mode == 'side' else 'custom_top'
        # This outputs a dict with key 'pixels'
        env = pixels.Wrapper(env, pixels_only=True, 
                             render_kwargs={'width': 84, 'height': 84, 'camera_id': cam_id})

    # 2. STANDARD: Fallback for normal DMC tasks
    else:
        domain, task = name.split('_', 1)
        if domain == 'cup': domain = 'ball_in_cup'

        env = suite.load(domain, task, task_kwargs={'random': seed}, visualize_reward=False)

        env = ActionDTypeWrapper(env, np.float32)
        env = ActionRepeatWrapper(env, action_repeat)
        env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

        # This outputs a dict with key 'pixels'
        env = pixels.Wrapper(env, pixels_only=True, render_kwargs={'width': 84, 'height': 84, 'camera_id': 0})
        env = ExtendedTimeStepWrapper(env)

    # 3. GLOBAL WRAPPERS
    # We pass pixels_key='pixels' because that is what pixels.Wrapper provides
    env = FrameStackWrapper(env, frame_stack, pixels_key='pixels')
    env = ExtendedTimeStepWrapper(env)
    env = ObservationDictWrapper(env) 
    
    return env