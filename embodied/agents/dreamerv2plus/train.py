import os
import pathlib
import pickle
import sys
import warnings
import ctypes

import numpy as np

# TensorFlow 2.16 + Keras 3 requires legacy tf.keras for TFP 0.23.
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
# Headless server defaults for dm_control rendering.
if 'DISPLAY' not in os.environ:
  os.environ.setdefault('MUJOCO_GL', 'egl')
  os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')


def _prefer_env_nvidia_libs():
  """Prefer CUDA/cuDNN libs from the active env over system-wide installs."""
  prefix = os.environ.get('CONDA_PREFIX')
  if not prefix:
    return
  pyver = f'python{sys.version_info.major}.{sys.version_info.minor}'
  base = pathlib.Path(prefix) / 'lib' / pyver / 'site-packages' / 'nvidia'
  libdirs = [
      base / 'cudnn' / 'lib',
      base / 'cublas' / 'lib',
      base / 'cuda_runtime' / 'lib',
      base / 'cuda_nvrtc' / 'lib',
      base / 'cusolver' / 'lib',
      base / 'cusparse' / 'lib',
      base / 'nccl' / 'lib',
  ]
  libdirs = [str(x) for x in libdirs if x.exists()]
  if libdirs:
    current = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = (
        ':'.join(libdirs + ([current] if current else [])))
  cudnn = base / 'cudnn' / 'lib' / 'libcudnn.so.8'
  if cudnn.exists():
    ctypes.CDLL(str(cudnn), mode=ctypes.RTLD_GLOBAL)


_prefer_env_nvidia_libs()

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')

directory = pathlib.Path(__file__)
try:
  import google3  # noqa
except ImportError:
  directory = directory.resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied


def main(argv=None):
  from . import agent as agnt

  parsed, other = embodied.Flags(
      configs=['defaults'], worker=0, workers=1,
      learner_addr='localhost:2222',
  ).parse_known(argv)
  config = embodied.Config(agnt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agnt.Agent.configs[name])
  # Backward-compatible alias for eval-only CLI override.
  other = [
      flag.replace('--eval.eps', '--train.eval_eps', 1)
      if flag == '--eval.eps' or flag.startswith('--eval.eps=')
      else flag
      for flag in other
  ]
  config = embodied.Flags(config).parse(other)

  config = config.update(logdir=str(embodied.Path(config.logdir)))
  args = embodied.Config(logdir=config.logdir, **config.train)
  args = args.update(expl_until=args.expl_until // config.env.repeat)
  print(config)

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()

  outdir = logdir
  multiplier = config.env.repeat
  if config.run == 'acting':
    outdir /= f'worker{parsed.worker}'
    multiplier *= parsed.workers
  elif config.run == 'learning':
    outdir /= 'learner'
    multiplier = 1
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(config.filter),
      embodied.logger.JSONLOutput(outdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(outdir, 'scores.jsonl', 'episode/score'),
      embodied.logger.TensorBoardOutput(outdir),
  ], multiplier)

  cleanup = []
  try:
    config = config.update({'env.seed': hash((config.seed, parsed.worker))})
    env = embodied.envs.load_env(
        config.task, mode='train', logdir=logdir, **config.env)
    agent = agnt.Agent(env.obs_space, env.act_space, step, config)
    cleanup.append(env)

    if config.run == 'train':
      replay = make_replay(config, logdir / 'episodes')
      embodied.run.train(agent, env, replay, logger, args)

    elif config.run == 'train_eval':
      eval_env = embodied.envs.load_env(
          config.task, mode='eval', logdir=logdir, **config.env)
      replay = make_replay(config, logdir / 'episodes')
      eval_replay = make_replay(config, logdir / 'eval_episodes', is_eval=True)
      embodied.run.train_eval(
          agent, env, eval_env, replay, eval_replay, logger, args)
      cleanup.append(eval_env)

    elif config.run == 'train_fixed_eval':
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        assert config.train.eval_fill
        eval_replay = make_replay(
            config, logdir / 'eval_episodes', is_eval=True)
      replay = make_replay(config, logdir / 'episodes')
      embodied.run.train_fixed_eval(
          agent, env, replay, eval_replay, logger, args)

    elif config.run == 'learning':
      env.close()
      port = parsed.learner_addr.split(':')[-1]
      replay = make_replay(config, logdir / 'episodes', server_port=port)
      if config.eval_dir:
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        eval_replay = replay
      embodied.run.learning(agent, replay, eval_replay, logger, args)

    elif config.run == 'acting':
      replay = make_replay(config, remote_addr=parsed.learner_addr)
      embodied.run.acting(agent, env, replay, logger, outdir, args)

    elif config.run == 'eval':
      eval_env = embodied.envs.load_env(
          config.task, mode='eval', logdir=logdir, **config.env)
      cleanup.append(eval_env)
      eval_replay_dir = (
          embodied.Path(config.eval_dir)
          if config.eval_dir else logdir / 'eval_episodes')
      eval_replay = make_replay(config, eval_replay_dir, is_eval=True)

      print('Loaded Eval Environment. Starting evaluation loop...')
      print(f'Eval episodes will be saved to: {eval_replay_dir}')

      driver = embodied.Driver(eval_env)
      driver.on_step(eval_replay.add)
      driver.on_episode(lambda ep, worker: print(f'Eval Episode {len(eval_replay)}: Score {ep["reward"].sum()}'))
      driver.on_episode(lambda ep, worker: logger.add(eval_replay.stats))
      driver.on_episode(lambda ep, worker: logger.write())

      init_obs = {
          k: np.zeros((len(eval_env),) + v.shape, v.dtype)
          for k, v in eval_env.obs_space.items()
      }
      if 'is_first' in init_obs:
        init_obs['is_first'] = np.ones(len(eval_env), bool)
      agent.policy(init_obs, mode='eval')

      checkpoint_path = logdir / 'checkpoint.pkl'
      print(f'Loading checkpoint: {checkpoint_path}')
      with checkpoint_path.open('rb') as f:
        checkpoint_data = pickle.load(f)
      if 'agent' not in checkpoint_data:
        raise KeyError("Missing 'agent' in checkpoint.")
      try:
        agent.load(checkpoint_data['agent'])
      except Exception as exc:
        print(f'Strict agent checkpoint load failed: {exc}')
        print('Retrying with non-strict tensor matching for eval.')
        agent.load(checkpoint_data['agent'], strict=False)
      if 'step' in checkpoint_data:
        step.load(checkpoint_data['step'])

      policy = lambda *args: agent.policy(*args, mode='eval')
      driver(policy, episodes=int(config.train.eval_eps))

    else:
      raise NotImplementedError(config.run)

  finally:
    for obj in cleanup:
      obj.close()


def make_replay(
    config, directory=None, is_eval=False, remote_addr=None,
    server_port=None, **kwargs):

  size = config.replay_size
  chunk = config.replay_chunk
  if is_eval:
    size //= 10
  if remote_addr:
    store = embodied.replay.StoreClient(remote_addr)
  else:
    if directory:
      store = embodied.replay.CkptRAMStore(
          directory, size, parallel=True)
    else:
      store = embodied.replay.RAMStore(size)
  store = embodied.replay.Stats(store)
  if server_port:
    store = embodied.replay.StoreServer(store, server_port)

  if config.replay == 'fixed':
    replay = embodied.replay.FixedLength(
        store, chunk, **config.replay_fixed, **kwargs)
  elif config.replay == 'consec':
    replay = embodied.replay.Consecutive(
        store, chunk, **config.replay_consec, **kwargs)
  elif config.replay == 'prio' and not is_eval:
    replay = embodied.replay.Prioritized(
        store, chunk, **config.replay_prio, **kwargs)
  elif config.replay == 'prio' and is_eval:
    # Don't use prioritized for eval.
    replay = embodied.replay.FixedLength(
        store, chunk, **config.replay_fixed, **kwargs)
  else:
    raise NotImplementedError(config.replay)
  return replay


if __name__ == '__main__':
  main()
