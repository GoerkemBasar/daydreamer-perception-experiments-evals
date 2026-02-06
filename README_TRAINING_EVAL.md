# DayDreamer Training & Evaluation Guide (Gruenau)

This guide documents the commands that are currently working for our custom
walker experiments (`side` and `top`) on Gruenau.

## 0. Reconnect to University SSH (If Session Closed)

From your local machine:

```bash
ssh <HU_USERNAME>@gruenau.informatik.hu-berlin.de
ssh gruenau10
```

Re-attach tmux:

```bash
tmux ls
tmux attach -t dreamer
```

If no session exists:

```bash
tmux new -s dreamer
```

### Safely Close tmux Session

If training/eval should keep running:

```bash
# Detach only (session keeps running)
# Press: Ctrl+b then d
```

If you want to fully stop and close:

```bash
# 1) Re-attach and stop running command with Ctrl+C in that pane
tmux attach -t dreamer

# 2) Exit shell in each pane/window
exit

# 3) Optional force-close session (only if you are sure)
tmux kill-session -t dreamer
```

## 1. Quick Start

```bash
conda activate dreamer
nvidia-smi
```

Notes:
- `train.py` now auto-handles headless rendering and CUDA/cuDNN library
  preference for this environment.
- You still need to `conda activate dreamer` before running commands.

## 2. Training Commands (Used Setup)

These flags must match at eval time (especially precision and cnn keys).

### Side-view training

```bash
python -u embodied/agents/dreamerv2plus/train.py \
  --run train \
  --logdir "logdir/run_gruenau_server_side_01" \
  --eval_dir "logdir/run_gruenau_server_side_01/eval_episodes" \
  --task dmc_custom_walker_side \
  --tf.platform gpu \
  --tf.jit True \
  --tf.precision float32 \
  --env.amount 1 \
  --env.parallel none \
  --encoder.cnn_keys '^image$' \
  --decoder.cnn_keys '^image$' \
  --batch_size 16 \
  --train.log_keys_video image \
  --train.steps 1000000 \
  --train.train_every 5 \
  --train.pretrain 100 \
  --train.log_every 100
```

### Top-view training

```bash
python -u embodied/agents/dreamerv2plus/train.py \
  --run train \
  --logdir "logdir/run_gruenau_server_top_01" \
  --eval_dir "logdir/run_gruenau_server_top_01/eval_episodes" \
  --task dmc_custom_walker_top \
  --tf.platform gpu \
  --tf.jit True \
  --tf.precision float32 \
  --env.amount 1 \
  --env.parallel none \
  --encoder.cnn_keys '^image$' \
  --decoder.cnn_keys '^image$' \
  --batch_size 16 \
  --train.log_keys_video image \
  --train.steps 1000000 \
  --train.train_every 5 \
  --train.pretrain 100 \
  --train.log_every 100
```

## 3. Standard Evaluation (Same World)

### Side brain in side world

```bash
python -u embodied/agents/dreamerv2plus/train.py \
  --run eval \
  --logdir "logdir/run_gruenau_server_side_01" \
  --eval_dir "logdir/run_gruenau_server_side_01/eval_episodes" \
  --task dmc_custom_walker_side \
  --env.amount 1 \
  --env.parallel none \
  --batch_size 1 \
  --eval.eps 10 \
  --tf.platform gpu \
  --tf.jit True \
  --tf.precision float32 \
  --encoder.cnn_keys '^image$' \
  --decoder.cnn_keys '^image$'
```

### Top brain in top world

```bash
python -u embodied/agents/dreamerv2plus/train.py \
  --run eval \
  --logdir "logdir/run_gruenau_server_top_01" \
  --eval_dir "logdir/run_gruenau_server_top_01/eval_episodes" \
  --task dmc_custom_walker_top \
  --env.amount 1 \
  --env.parallel none \
  --batch_size 1 \
  --eval.eps 10 \
  --tf.platform gpu \
  --tf.jit True \
  --tf.precision float32 \
  --encoder.cnn_keys '^image$' \
  --decoder.cnn_keys '^image$'
```

## 4. Cross-Evaluation ("Blindfold Test")

Use separate `eval_dir` folders to keep cross results isolated.

### Top brain in side world

```bash
python -u embodied/agents/dreamerv2plus/train.py \
  --run eval \
  --logdir "logdir/run_gruenau_server_top_01" \
  --eval_dir "logdir/run_gruenau_server_top_01/cross_eval_side_world_episodes" \
  --task dmc_custom_walker_side \
  --env.amount 1 \
  --env.parallel none \
  --batch_size 1 \
  --eval.eps 10 \
  --tf.platform gpu \
  --tf.jit True \
  --tf.precision float32 \
  --encoder.cnn_keys '^image$' \
  --decoder.cnn_keys '^image$'
```

### Side brain in top world

```bash
python -u embodied/agents/dreamerv2plus/train.py \
  --run eval \
  --logdir "logdir/run_gruenau_server_side_01" \
  --eval_dir "logdir/run_gruenau_server_side_01/cross_eval_top_world_episodes" \
  --task dmc_custom_walker_top \
  --env.amount 1 \
  --env.parallel none \
  --batch_size 1 \
  --eval.eps 10 \
  --tf.platform gpu \
  --tf.jit True \
  --tf.precision float32 \
  --encoder.cnn_keys '^image$' \
  --decoder.cnn_keys '^image$'
```

## 5. Check That Eval Episodes Were Saved

```bash
find logdir/run_gruenau_server_side_01/eval_episodes -maxdepth 1 -name '*.npz' | wc -l
find logdir/run_gruenau_server_top_01/eval_episodes -maxdepth 1 -name '*.npz' | wc -l
find logdir/run_gruenau_server_top_01/cross_eval_side_world_episodes -maxdepth 1 -name '*.npz' | wc -l
find logdir/run_gruenau_server_side_01/cross_eval_top_world_episodes -maxdepth 1 -name '*.npz' | wc -l
```

## 6. Convert `.npz` Episodes to Video (`.mp4`)

### Convert the latest episode in a folder

```bash
python - <<'PY'
import glob
import numpy as np
import imageio.v2 as imageio

folder = "logdir/run_gruenau_server_side_01/eval_episodes"  # change as needed
files = sorted(glob.glob(f"{folder}/*.npz"))
if not files:
    raise SystemExit("No npz files found.")

episode = files[-1]
with np.load(episode) as data:
    frames = data['image']
imageio.mimsave("latest_eval.mp4", frames, fps=60)
print("Saved latest_eval.mp4 from", episode)
PY
```

### Convert the best episode (highest return) in a folder

```bash
python - <<'PY'
import glob
import numpy as np
import imageio.v2 as imageio

folder = "logdir/run_gruenau_server_side_01/eval_episodes"  # change as needed
files = sorted(glob.glob(f"{folder}/*.npz"))
if not files:
    raise SystemExit("No npz files found.")

def score(path):
    with np.load(path) as d:
        return float(d['reward'].sum())

best = max(files, key=score)
with np.load(best) as d:
    imageio.mimsave("best_eval.mp4", d['image'], fps=60)
print("Saved best_eval.mp4 from", best, "score=", score(best))
PY
```
