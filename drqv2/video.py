# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import imageio
import numpy as np

class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20, camera_id=0):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id
        
        # DEBUG PRINT: Verify initialization
        print(f"\n[VideoRecorder] Initialized with camera_id: '{self.camera_id}'\n")

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                # DEBUG PRINT: Verify what is actually being passed to render
                # print(f"[VideoRecorder] Rendering frame with camera_id: {self.camera_id}")
                
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=self.camera_id)
            else:
                frame = env.render()
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)

class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20, camera_id=0):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            # We assume obs is in channel-first format (C, H, W)
            # We take the last 3 channels (RGB) and transpose to (H, W, C) for cv2
            if isinstance(obs, dict):
                obs = obs['image']
            
            # Safety check: ensure obs is on CPU numpy
            if hasattr(obs, 'cpu'):
                obs = obs.cpu().numpy()
                
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)