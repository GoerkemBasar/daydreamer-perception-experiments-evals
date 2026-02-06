import os
import functools
import types  # Required for monkey patching

import embodied
import numpy as np
from dm_control import suite
from dm_control import mjcf
from dm_control.rl import control
from dm_control.suite import walker as walker_module
from dm_control.suite.wrappers import pixels

class DMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
      locom_rodent_maze_forage=1,
      locom_rodent_two_touch=1,
      quadruped_escape=2,
      quadruped_fetch=2,
      quadruped_run=2,
      quadruped_walk=2,
  )

  def __init__(self, name, repeat=1, size=(64, 64), camera=-1):
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'

    self._repeat = repeat
    self._size = size

    # -------------------------------------------------------------------------
    # CUSTOM WALKER LOGIC (Ported from your working DrQ-v2 setup)
    # -------------------------------------------------------------------------
    if isinstance(name, str) and name.startswith('custom_walker'):
        view_mode = name.split('_')[-1]
        
        # 1. Load XML
        walker_dir = os.path.dirname(walker_module.__file__)
        xml_path = os.path.join(walker_dir, 'walker.xml')
        root = mjcf.from_path(xml_path)
        
        # 2. Inject Camera
        if view_mode == 'top':
            root.worldbody.add('camera', name='custom_camera', 
                               mode='trackcom', target='torso',
                               pos=[0, 0, 4], xyaxes=[0, -1, 0, 1, 0, 0])
        else:
            # Default to Side view
            root.worldbody.add('camera', name='custom_camera', 
                               mode='trackcom', target='torso',
                               pos=[0, -3, 1], xyaxes=[1, 0, 0, 0, 0, 1])
        
        # 3. Compile Physics
        physics = mjcf.Physics.from_mjcf_model(root)

        # 4. MONKEY PATCHING (The Secret Sauce)
        # This manually adds the missing methods that caused the crashes.
        def horizontal_velocity():
            return physics.data.qvel[0]

        def orientations():
            return physics.data.qpos[3:]

        def torso_height():
            return physics.named.data.xpos['torso', 'z']

        def torso_upright():
            return physics.named.data.xmat['torso', 'zz']

        # Bind these functions to the physics instance
        physics.horizontal_velocity = horizontal_velocity
        physics.orientations = orientations
        physics.torso_height = torso_height
        physics.torso_upright = torso_upright

        # 5. Create Task
        # We assume seed 0 for now since 'seed' isn't passed to __init__ in DayDreamer typically
        # (It's handled by the environment wrapper later)
        task = walker_module.PlanarWalker(move_speed=walker_module._WALK_SPEED, random=0)
        
        # 6. Build Environment
        env = control.Environment(physics, task, time_limit=25)
        
        # 7. Apply Pixel Wrapper (RGB Only)
        self._env = pixels.Wrapper(
            env, 
            pixels_only=True, 
            render_kwargs={'width': 64, 'height': 64, 'camera_id': 'custom_camera'}
        )
        self._camera = 'custom_camera'

    # -------------------------------------------------------------------------
    # STANDARD LOGIC (Original DayDreamer Code)
    # -------------------------------------------------------------------------
    else:
        if not isinstance(name, str):
          self._env = name
        else:
          domain, task = name.split('_', 1)
          if domain == 'cup': domain = 'ball_in_cup'
          if domain == 'manip':
            from dm_control import manipulation
            self._env = manipulation.load(task + '_vision')
          elif domain == 'locom':
            from dm_control.locomotion.examples import basic_rodent_2020
            self._env = getattr(basic_rodent_2020, task)()
          else:
            self._env = suite.load(domain, task)
        
        if camera == -1:
          camera = self.DEFAULT_CAMERAS.get(name, 0)
        self._camera = camera

    # -------------------------------------------------------------------------
    # COMMON SETUP
    # -------------------------------------------------------------------------
    self._ignored_keys = []
    for key, value in self._env.observation_spec().items():
      if value.shape == (0,):
        print(f"Ignoring empty observation key '{key}'.")
        self._ignored_keys.append(key)
    self._done = True

  @property
  def obs_space(self):
    spaces = {
        'image': embodied.Space(np.uint8, self._size + (3,)),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }
    for key, value in self._env.observation_spec().items():
      if key in self._ignored_keys:
        continue
      shape = (1,) if value.shape == () else value.shape
      if np.issubdtype(value.dtype, np.floating):
        spaces[key] = embodied.Space(np.float32, shape)
      elif np.issubdtype(value.dtype, np.uint8):
        spaces[key] = embodied.Space(np.uint8, shape)
      else:
        raise NotImplementedError(value.dtype)
    return spaces

  @property
  def act_space(self):
    spec = self._env.action_spec()
    return {
        'action': embodied.Space(np.float32, None, spec.minimum, spec.maximum),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      time_step = self._env.reset()
      self._done = False
      return self._obs(time_step, 0.0)
    assert np.isfinite(action['action']).all(), action['action']
    reward = 0.0
    for _ in range(self._repeat):
      time_step = self._env.step(action['action'])
      reward += time_step.reward or 0.0
      if time_step.last():
        break
    assert time_step.discount in (0, 1)
    self._done = time_step.last()
    return self._obs(time_step, reward)

  def _obs(self, time_step, reward):
    # Convert to dict to allow popping
    raw_obs = dict(time_step.observation)
    
    # Keep both keys for compatibility: some configs expect `image`, others
    # consume `pixels` directly.
    if 'pixels' in raw_obs and 'image' not in raw_obs:
        raw_obs['image'] = raw_obs['pixels']

    obs = {
        k: v[None] if v.shape == () else v
        for k, v in raw_obs.items()
        if k not in self._ignored_keys}
    
    data = dict(
        reward=reward,
        is_first=time_step.first(),
        is_last=time_step.last(),
        is_terminal=time_step.discount == 0,
        **obs,
    )
    
    # Fallback render if image is missing
    if 'image' not in data and hasattr(self, 'render'):
         data['image'] = self.render()
         
    return data

  def render(self):
    return self._env.physics.render(*self._size, camera_id=self._camera)
