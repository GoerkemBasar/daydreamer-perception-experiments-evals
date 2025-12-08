from dm_control import suite
from dm_control import mjcf
from dm_control.suite.wrappers import pixels
import numpy as np

def make_custom_walker(task_name='walk', view='side', seed=1):
    """
    Creates the Walker environment with a specific camera view and RGB-only observations.
    
    Args:
        task_name (str): 'walk' or 'run' or 'stand'
        view (str): 'side' (Easy) or 'top' (Hard)
        seed (int): Random seed
    """
    # 1. Load the Standard Environment
    env = suite.load(domain_name='walker', task_name=task_name, task_kwargs={'random': seed})
    
    # 2. Inject Custom Cameras
    # We modify the MJCF model directly before the physics engine is fully locked in
    arena = env.physics.model
    
    if view == 'side':
        # Standard side view (Easy Baseline)
        # Position: y=-3 (back), z=1 (waist height). xyaxes setup points camera at robot.
        arena.worldbody.add('camera', 
                            name='custom_view',
                            mode='trackcom',
                            target = 'torso',
                            pos=[0, -3, 1], xyaxes=[1, 0, 0, 0, 0, 1]) 
    
    elif view == 'top':
        # Top-down bird's eye view (Hard/Ambiguous)
        # Position: z=4 (high up). Pointing straight down.
        arena.worldbody.add('camera', 
                            name='custom_view', 
                            mode='trackcom',
                            target = 'torso',
                            pos=[0, 0, 4], xyaxes=[0, -1, 0, 1, 0, 0])
        
    else:
        raise ValueError(f"Unknown view: {view}. Use 'side' or 'top'.")

    # 3. Apply Pixel Wrapper (CRITICAL: pixels_only=True)
    # This removes proprioception (joint angles), forcing the agent to use the image.
    # resolution 64x64 is standard for Dreamer; 84x84 is standard for DrQ-v2.
    # We will let the algorithm wrappers handle resizing if needed, but 64x64 is a safe base.
    env = pixels.Wrapper(env, 
                         pixels_only=True, 
                         render_kwargs={'height': 64, 'width': 64, 'camera_id': 'custom_view'})
    
    return env