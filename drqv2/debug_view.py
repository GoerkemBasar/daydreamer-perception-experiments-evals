import os
import cv2  # You likely have opencv-python installed, or use PIL
import numpy as np
from dm_control import mjcf, suite
from dm_control.suite import walker

# --- REPLICATING YOUR LOGIC ---
def make_custom_env(view_mode):
    print(f"--- Building Env for {view_mode} View ---")
    
    # 1. Load Standard Walker XML
    walker_dir = os.path.dirname(walker.__file__)
    xml_path = os.path.join(walker_dir, 'walker.xml')
    root = mjcf.from_path(xml_path)
    
    # 2. Inject Camera
    if view_mode == 'top':
        print("Injecting TOP camera...")
        root.worldbody.add('camera', name='custom_top', 
                           mode='trackcom', target='torso',
                           pos=[0, 0, 4], xyaxes=[0, -1, 0, 1, 0, 0])
        cam_name = 'custom_top'
    else:
        print("Injecting SIDE camera...")
        root.worldbody.add('camera', name='custom_side', 
                           mode='trackcom', target='torso',
                           pos=[0, -3, 1], xyaxes=[1, 0, 0, 0, 0, 1])
        cam_name = 'custom_side'

    # 3. Build Physics
    physics = mjcf.Physics.from_mjcf_model(root)

    # 4. FIX: Use PlanarWalker, not Walk
    task = walker.PlanarWalker(move_speed=walker._WALK_SPEED, random=1)
    
    # 5. Render
    # Force a reset to initialize physics
    task.initialize_episode(physics)
    
    # Render the specific camera
    pixels = physics.render(height=240, width=240, camera_id=cam_name)
    return pixels

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Ensure we use GLFW to avoid hanging
    os.environ['MUJOCO_GL'] = 'glfw' 
    
    try:
        # Generate Top View
        top_pixels = make_custom_env('top')
        # Convert RGB to BGR for OpenCV saving (or just save if using PIL)
        cv2.imwrite('debug_top_view.png', cv2.cvtColor(top_pixels, cv2.COLOR_RGB2BGR))
        print("SUCCESS: Saved 'debug_top_view.png'")

        # Generate Side View
        side_pixels = make_custom_env('side')
        cv2.imwrite('debug_side_view.png', cv2.cvtColor(side_pixels, cv2.COLOR_RGB2BGR))
        print("SUCCESS: Saved 'debug_side_view.png'")
        
        print("\nCHECK YOUR FOLDER NOW. If 'debug_top_view.png' looks like a map, it works.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")