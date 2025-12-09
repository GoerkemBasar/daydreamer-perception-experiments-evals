# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control import mjcf
from dm_control.rl import control
from dm_control.suite import walker
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

# --- CUSTOM WRAPPERS ---

class TimeLimitWrapper(dm_env.Environment):
    """Forces the environment to terminate after a fixed number of steps."""
    def __init__(self, env, max_steps=1000):
        self._env = env
        self._max_steps = max_steps
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        return self._env.reset()

    def step(self, action):
        time_step = self._env.step(action)
        self._step_count += 1
        
        if self._step_count >= self._max_steps and not time_step.last():
            # Force termination
            return time_step._replace(step_type=StepType.LAST)
        
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

def make_custom_walker_direct(view, seed):
    # 1. Load XML
    walker_dir = os.path.dirname(walker.__file__)
    xml_path = os.path.join(walker_dir, 'walker.xml')
    root = mjcf.from_path(xml_path)
    
    # 2. Inject Camera
    if view == 'top':
        root.worldbody.add('camera', name='custom_top', 
                           mode='trackcom', target='torso',
                           pos=[0, 0, 4], xyaxes=[0, -1, 0, 1, 0, 0])
    else:
        root.worldbody.add('camera', name='custom_side', 
                           mode='trackcom', target='torso',
                           pos=[0, -3, 1], xyaxes=[1, 0, 0, 0, 0, 1])
    
    # 3. Compile Physics
    physics = mjcf.Physics.from_mjcf_model(root)

    # --- MONKEY PATCHING ---
    def horizontal_velocity():
        return physics.data.qvel[0]

    def orientations():
        return physics.data.qpos[3:]

    def torso_height():
        return physics.named.data.xpos['torso', 'z']

    def torso_upright():
        return physics.named.data.xmat['torso', 'zz']

    physics.horizontal_velocity = horizontal_velocity
    physics.orientations = orientations
    physics.torso_height = torso_height
    physics.torso_upright = torso_upright
    # -----------------------

    # 4. Task
    task = walker.PlanarWalker(move_speed=walker._WALK_SPEED, random=seed)
    
    # 5. Build Env (time_limit=25 is set here, but we will add a wrapper to be safe)
    env = control.Environment(physics, task, time_limit=25)
    
    return env

# --- STANDARD WRAPPERS ---

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
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [pixels_shape[:2], [pixels_shape[2] * num_frames]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='image')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=2)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
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

# --- MAIN MAKE FUNCTION ---

def make(name, frame_stack, action_repeat, seed):
    
    # >>> CASE 1: CUSTOM WALKER <<<
    if name.startswith('custom_walker'):
        view_mode = name.split('_')[-1] 
        print(f"\nDEBUG: Initializing Custom Walker. View Mode: {view_mode}")

        env = make_custom_walker_direct(view=view_mode, seed=seed)
        
        # Metadata
        env.domain_name = 'walker'
        env.task_name = view_mode

        # WRAPPERS
        # 1. Force Time Limit (Fixes infinite episode issue)
        # 1000 steps * 0.025s = 25s
        env = TimeLimitWrapper(env, max_steps=1000)

        env = ActionDTypeWrapper(env, np.float32)
        env = ActionRepeatWrapper(env, action_repeat)
        env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

        cam_id = 'custom_side' if view_mode == 'side' else 'custom_top'
        print(f"DEBUG: Selected Camera ID: {cam_id}")
        
        env = pixels.Wrapper(env, pixels_only=True, 
                             render_kwargs={'width': 84, 'height': 84, 'camera_id': cam_id})

    # >>> CASE 2: STANDARD DMC <<<
    else:
        domain, task = name.split('_', 1)
        if domain == 'cup': domain = 'ball_in_cup'

        env = suite.load(domain, task, task_kwargs={'random': seed}, visualize_reward=False)

        env = ActionDTypeWrapper(env, np.float32)
        env = ActionRepeatWrapper(env, action_repeat)
        env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

        env = pixels.Wrapper(env, pixels_only=True, render_kwargs={'width': 84, 'height': 84, 'camera_id': 0})
        env = ExtendedTimeStepWrapper(env)

    # >>> GLOBAL WRAPPERS <<<
    env = FrameStackWrapper(env, frame_stack, pixels_key='pixels')
    env = ExtendedTimeStepWrapper(env)
    env = ObservationDictWrapper(env) 
    
    return env