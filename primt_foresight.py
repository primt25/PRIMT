import numpy as np

from gym.wrappers.time_limit import TimeLimit
from rlkit.envs.wrappers import NormalizedBoxEnv
import metaworld.envs.mujoco.env_dict as _env_dict

from primt_fm import (FORESIGHT_SYSTEM, FORESIGHT_USER, parse_code_block,
                      parse_strategy_samples)
from primt_projection import get_task_description
from primt_render import pack_env_state

TRAJECTORY_FORMAT = """1. [0:2]  - End-effector (TCP) position in Cartesian space, in meters
2. [3]    - Gripper opening level: 0 = closed, 1 = open
3. [4:6]  - Object position (button), 3D, in meters
4. [7:10] - Object orientation (quaternion, 4D)
5. [11:13]- TCP delta action (dx, dy, dz)
6. [14]   - Gripper torque control (normalized)"""

TRAJECTORY_CONSTRAINTS = """- Trajectory must contain exactly {n_steps} steps
- Each transition must be smooth (no large discontinuities between consecutive steps)
- Avoid collisions with the table (assumed z > 0.05)
- Action values [11:14] (TCP deltas and gripper torque) must be consistent with state changes in [0:3] and [3]"""

ENVIRONMENT_SETUP = """- Robot: Sawyer arm with a top-down gripper
- Workspace frame:
  - x-axis: left -> right
  - y-axis: near -> far
  - z-axis: bottom -> top
- The button faces the robot and is pressed by pushing it along the +y direction.
- Gripper logic: the gripper is not needed for this task; keep it open (1.0).
- Initial values provided at run time in init_state:
  - tcp: initial TCP position (3D), around [0.0, 0.4, 0.2]
  - obj_pos: initial button position (3D), x in [-0.1, 0.1], y ~ 0.68, z ~ 0.12
  - obj_ori: initial object orientation (4D quaternion)"""

CODE_RETURN_FORMAT = '''```python
def generate_trajectory(init_state, strategy_config, env_params):
    """
    Generate a full trajectory by sequentially composing functional
    motion phases.

    Args:
        init_state (dict): 'tcp' (3,), 'grip' float, 'obj_pos' (3,),
            'obj_ori' (4,)
        strategy_config (dict): trajectory-level motion parameters
        env_params (dict): task constants, includes 'n_steps'

    Returns:
        trajectory (np.ndarray): array of shape (n_steps, 15)
    """
    trajectory = []
    # === Phase 1 ===
    # traj_phase1 = phase_xxx(...)
    # trajectory += traj_phase1
    # ...
    return np.stack(trajectory, axis=0)
```
(You may define helper primitives such as phase_approach / linear_segment /
build_step. Only use numpy.)'''


def _make_env(env_name, seed):
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]
    env = env_cls()
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)
    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)


def _exec_generated_code(code):
    import math

    def _safe_import(name, *args, **kwargs):
        if name in ('numpy', 'math', 'numpy.linalg'):
            return __import__(name, *args, **kwargs)
        raise ImportError(f'import of {name!r} not allowed in generated code')

    ns = {'np': np, 'numpy': np, 'math': math, '__builtins__': {
        'range': range, 'len': len, 'min': min, 'max': max, 'abs': abs,
        'float': float, 'int': int, 'list': list, 'dict': dict,
        'tuple': tuple, 'enumerate': enumerate, 'zip': zip, 'print': print,
        'sum': sum, 'round': round, 'sorted': sorted, 'reversed': reversed,
        'isinstance': isinstance, 'Exception': Exception,
        'ValueError': ValueError, '__import__': _safe_import,
        '__build_class__': __builtins__['__build_class__']
        if isinstance(__builtins__, dict)
        else getattr(__builtins__, '__build_class__'),
        '__name__': 'primt_foresight_gen'}}
    exec(code, ns)
    fn = ns.get('generate_trajectory')
    if fn is None:
        raise ValueError('generated code defines no generate_trajectory()')
    return fn


def request_foresight_program(fm, env_name, num_samples, n_steps,
                              code_retries=3):
    task_desc = get_task_description(env_name)
    prompt = FORESIGHT_USER.format(
        task_description=task_desc,
        trajectory_format=TRAJECTORY_FORMAT,
        trajectory_constraints=TRAJECTORY_CONSTRAINTS.format(n_steps=n_steps),
        environment_setup=ENVIRONMENT_SETUP,
        code_return_format=CODE_RETURN_FORMAT,
        n=num_samples)
    messages = [{'role': 'system', 'content': FORESIGHT_SYSTEM},
                {'role': 'user', 'content': prompt}]
    last_err = None
    for attempt in range(code_retries):
        try:
            resp = fm.chat_messages(messages, salt=f'foresight_a{attempt}',
                                    kind='foresight')
        except Exception as e:
            last_err = e
            continue
        code = parse_code_block(resp)
        samples = parse_strategy_samples(resp)
        try:
            if code is None:
                raise ValueError('no python code block in response')
            fn = _exec_generated_code(code)
            test = fn({'tcp': np.array([0.0, 0.4, 0.2]), 'grip': 1.0,
                       'obj_pos': np.array([0.0, 0.68, 0.12]),
                       'obj_ori': np.array([1.0, 0.0, 0.0, 0.0])},
                      samples[0].get('strategy_config', {})
                      if samples else {},
                      {'n_steps': n_steps})
            test = np.asarray(test)
            if test.ndim != 2 or test.shape[1] < 15:
                raise ValueError(f'bad trajectory shape {test.shape}')
            return fn, samples
        except Exception as e:
            last_err = e
            messages = messages[:2] + [
                {'role': 'assistant', 'content': resp},
                {'role': 'user', 'content':
                    f'Executing your code failed with: {type(e).__name__}: '
                    f'{e}\nPlease fix the code and reply again with the '
                    'full corrected Step 1-3 outputs in the same format.'}]
    print(f'[PRIMT foresight] code generation failed: {last_err}')
    return None, []


def _track_waypoints(env, waypoints, kp=12.0):
    obs = env.reset()
    T = env._max_episode_steps
    n_wp = len(waypoints)
    obs_l, act_l, rew_l, state_l = [], [], [], []
    inner = env
    while hasattr(inner, 'env'):
        inner = inner.env
    done, t = False, 0
    while not done:
        wp = waypoints[min(int(t * n_wp / T), n_wp - 1)]
        tcp_now = obs[0:3]
        act = np.zeros(env.action_space.shape[0], dtype=np.float32)
        act[:3] = np.clip(kp * (wp[0:3] - tcp_now), -1.0, 1.0)
        act[3] = np.clip(wp[14] if abs(wp[14]) > 1e-6 else
                         (wp[3] * 2.0 - 1.0) * -1.0, -1.0, 1.0)
        state_l.append(pack_env_state(inner.get_env_state(), inner))
        next_obs, reward, done, extra = env.step(act)
        obs_l.append(obs)
        act_l.append(act)
        rew_l.append([reward])
        obs = next_obs
        t += 1
    return obs_l, act_l, rew_l, state_l


def generate_foresight_trajectories(fm, cfg_env, num_traj, seed, rng=None):
    if rng is None:
        rng = np.random.RandomState(seed)
    env_name = cfg_env.replace('metaworld_', '')
    env = _make_env(env_name, seed + 777)
    T = env._max_episode_steps

    fn, samples = request_foresight_program(fm, cfg_env, num_traj, T)
    trajs = []
    for i in range(num_traj):
        obs0 = env.reset()
        init_state = {'tcp': obs0[0:3].copy(), 'grip': float(obs0[3]),
                      'obj_pos': obs0[4:7].copy(),
                      'obj_ori': obs0[7:11].copy()}
        waypoints = None
        if fn is not None:
            cfg_s = samples[i % len(samples)].get('strategy_config', {}) \
                if samples else {}
            cfg_s = {k: (v * float(rng.uniform(0.8, 1.2))
                         if isinstance(v, (int, float)) else v)
                     for k, v in cfg_s.items()}
            try:
                waypoints = np.asarray(fn(init_state, cfg_s,
                                          {'n_steps': T}), dtype=np.float64)
            except Exception as e:
                print(f'[PRIMT foresight] sample {i} failed: {e}')
        if waypoints is None or len(waypoints) < 2:
            waypoints = np.zeros((T, 15))
            waypoints[:, 0:3] = obs0[0:3]
            waypoints[:, 3] = 1.0
        obs_l, act_l, rew_l, state_l = _track_waypoints(env, waypoints)
        wide = np.concatenate(
            [np.asarray(obs_l, dtype=np.float32),
             np.asarray(act_l, dtype=np.float32),
             np.asarray(state_l, dtype=np.float32)], axis=-1)
        trajs.append((wide, np.array(rew_l, dtype=np.float32)))
    env.close()
    return trajs


def inject_foresight_into_reward_model(reward_model, fm, cfg_env, num_traj,
                                       seed):
    trajs = generate_foresight_trajectories(fm, cfg_env, num_traj, seed)
    for sa, r in trajs:
        reward_model.inputs.append(sa)
        reward_model.targets.append(r)
    reward_model.inputs.append([])
    reward_model.targets.append([])
    return len(trajs)


_SCRIPTED_POLICIES = {
    'button-press-v2': ('metaworld.policies.sawyer_button_press_v2_policy',
                        'SawyerButtonPressV2Policy'),
    'door-open-v2': ('metaworld.policies.sawyer_door_open_v2_policy',
                     'SawyerDoorOpenV2Policy'),
    'sweep-into-v2': ('metaworld.policies.sawyer_sweep_into_v2_policy',
                      'SawyerSweepIntoV2Policy'),
}


def _get_scripted_policy(env_name):
    for key, (module, cls) in _SCRIPTED_POLICIES.items():
        if env_name.startswith(key.split('-v')[0]):
            mod = __import__(module, fromlist=[cls])
            return getattr(mod, cls)()
    raise NotImplementedError(
        f'No scripted foresight policy for {env_name!r}. Supported: '
        f'{sorted(_SCRIPTED_POLICIES)}. Use primt.fm.backend=openai for the '
        'LLM code-generation path, or add a policy here.')


def generate_foresight_scripted(cfg_env, num_traj, seed, rng=None):
    if rng is None:
        rng = np.random.RandomState(seed)
    env_name = cfg_env.replace('metaworld_', '')
    env = _make_env(env_name, seed + 777)
    policy = _get_scripted_policy(env_name)
    inner = env
    while hasattr(inner, 'env'):
        inner = inner.env
    trajs = []
    for i in range(num_traj):
        act_scale = rng.uniform(0.25, 1.0)
        noise_std = rng.uniform(0.02, 0.35)
        delay = rng.randint(0, 60)
        stop_frac = rng.uniform(0.5, 1.0)
        T = env._max_episode_steps
        stop_step = int(T * stop_frac)
        obs = env.reset()
        obs_l, act_l, rew_l, state_l = [], [], [], []
        done, t = False, 0
        while not done:
            if t < delay or t >= stop_step:
                action = np.zeros(env.action_space.shape[0])
            else:
                action = policy.get_action(obs)
                action = action * act_scale + rng.randn(len(action)) * noise_std
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            state_l.append(pack_env_state(inner.get_env_state(), inner))
            next_obs, reward, done, extra = env.step(action)
            obs_l.append(obs)
            act_l.append(action)
            rew_l.append([reward])
            obs = next_obs
            t += 1
        wide = np.concatenate(
            [np.asarray(obs_l, dtype=np.float32),
             np.asarray(act_l, dtype=np.float32),
             np.asarray(state_l, dtype=np.float32)], axis=-1)
        trajs.append((wide, np.array(rew_l, dtype=np.float32)))
    env.close()
    return trajs


def inject_foresight_scripted(reward_model, cfg_env, num_traj, seed):
    trajs = generate_foresight_scripted(cfg_env, num_traj, seed)
    for sa, r in trajs:
        reward_model.inputs.append(sa)
        reward_model.targets.append(r)
    reward_model.inputs.append([])
    reward_model.targets.append([])
    return len(trajs)
