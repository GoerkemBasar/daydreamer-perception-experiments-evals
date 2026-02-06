import os
import glob
import numpy as np
import imageio

# --- CONFIGURATION (EDIT THIS) ---
# 1. Paste your specific file path here (inside the quotes). 
#    If you leave it empty "", the script will find the latest file automatically.
#MANUAL_FILE_PATH = "/vol/fob-vol6/mi25/basarmeh/daydreamer/logdir/run_gruenau_fixed/eval_episodes/20251213T114618-dafc58fff6c74773955e751cc5a3956e-len5001-rew534.npz"
MANUAL_FILE_PATH = "/vol/fob-vol6/mi25/basarmeh/daydreamer/logdir/run_gruenau_server_01/episodes/20260205T071928-9d58e330eac34d2fa3d711038ba2cb02-len10001-rew2813.npz"
# 2. Speed up the video? 
#    30 = Slow/Normal
#    60 = Fast (Real-time for many robots)
#    100 = Very Fast
PLAYBACK_FPS = 60
# ---------------------------------

def get_file():
    # If user provided a specific path, use it
    if MANUAL_FILE_PATH and os.path.exists(MANUAL_FILE_PATH):
        return MANUAL_FILE_PATH
    
    # Otherwise, find the latest one automatically
    search_pattern = 'logdir/run_gruenau_fixed/eval_episodes/*.npz' 
    # NOTE: Change 'eval_episodes' to 'episodes' above if you want training data
    
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        raise FileNotFoundError("No .npz files found! Check your path or wait for training.")
    return max(list_of_files, key=os.path.getctime)

def main():
    try:
        target_file = get_file()
        print(f"--> Loading: {target_file}")

        with open(target_file, 'rb') as f:
            data = np.load(f)
            # Smart logic to find the image key
            keys = list(data.keys())
            if 'image' in data:
                images = data['image']
            elif 'observation/image' in data:
                images = data['observation/image']
            else:
                # Find any key containing "image"
                image_key = next((k for k in keys if 'image' in k), None)
                if not image_key:
                    print(f"Error: Could not find images in file. Keys found: {keys}")
                    return
                images = data[image_key]

        output_filename = "custom_video.mp4"
        print(f"--> Rendering {len(images)} frames at {PLAYBACK_FPS} FPS...")
        
        imageio.mimsave(output_filename, images, fps=PLAYBACK_FPS)
        print(f"SUCCESS! Saved to: {output_filename}")
        print(f"Download it with: scp basarmeh@gruenau10.informatik.hu-berlin.de:~/daydreamer/{output_filename} ~/Desktop/")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()