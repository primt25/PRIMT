import re

import numpy as np
from scipy.ndimage import uniform_filter1d

TASK_DESCRIPTIONS = {
    'button-press-v2': 'to press a button on a surface',
    'door-open-v2': 'to open a door with a revolving joint',
    'sweep-into-v2': 'to sweep the object into the target area',
    'PickSingleYCB-v0': 'to pick up a random object sampled from the YCB '
                        'dataset and move it to a random goal position',
    'StackCube-v0': 'to pick up a red cube and stack it on top of a green '
                    'cube and let go of the cube without it falling',
    'PegInsertionSide-v0': 'to pick up an orange-white peg and insert the '
                           'orange end into the box with a hole in it',
    'hopper_stand': 'to stabilize a planar one-legged hopper initialized in a '
                    'random pose, encouraging upright posture with minimal '
                    'torso height loss',
    'walker_walk': 'to control a planar bipedal walker to move forward with a '
                   'target velocity',
}

KEYFRAME_PARAMS = {
    'metaworld': dict(delta_v=0.005, smooth_k=2, smooth_topk=5,
                      delta_e=0.01, cpd_penalty=20),
    'maniskill': dict(delta_v=0.025, smooth_k=4, smooth_topk=8,
                      delta_e=0.02, cpd_penalty=30),
    'dmc': dict(delta_v=0.065, smooth_k=6, smooth_topk=10,
                delta_e=0.04, cpd_penalty=40),
}

_BASE_CHANNELS = [
    ('tcp_x', 'obs', 0),
    ('tcp_y', 'obs', 1),
    ('tcp_z', 'obs', 2),
    ('gripper_open', 'obs', 3),
    ('obj_x', 'obs', 4),
    ('obj_y', 'obs', 5),
    ('obj_z', 'obs', 6),
]

_TARGET_CHANNELS = [
    ('target_x', 'obs', 36),
    ('target_y', 'obs', 37),
    ('target_z', 'obs', 38),
]

_ACTION_CHANNELS = [
    ('action_dx', 'act', 0),
    ('action_dy', 'act', 1),
    ('action_dz', 'act', 2),
    ('action_gripper', 'act', 3),
]

_CHANNEL_DOC = [
    ('tcp_x', '- tcp_x / tcp_y / tcp_z: robot end-effector (TCP) position in '
              'meters'),
    ('gripper_open', '- gripper_open: gripper opening level (0 = closed, '
                     '1 = open)'),
    ('obj_x', '- obj_x / obj_y / obj_z: manipulated object position in '
              'meters'),
    ('target_x', '- target_x / target_y / target_z: task goal (target) '
                 'position in meters'),
    ('action_dx', '- action_dx / action_dy / action_dz: commanded TCP delta '
                  'action, normalized to [-1, 1]'),
    ('action_gripper', '- action_gripper: gripper control command, normalized '
                       'to [-1, 1]'),
]


def get_task_description(env_name):
    key = env_name.replace('metaworld_', '').replace('maniskill_', '')
    return TASK_DESCRIPTIONS.get(key, 'to complete the manipulation task')


def get_keyframe_params(env_name, override=None):
    name = env_name.lower()
    if 'metaworld' in name:
        bench = 'metaworld'
    elif 'maniskill' in name:
        bench = 'maniskill'
    else:
        bench = 'dmc'
    params = dict(KEYFRAME_PARAMS[bench])
    if override:
        params.update({k: v for k, v in override.items() if v is not None})
    return params


def get_text_channels(ds, da):
    channels = [c for c in _BASE_CHANNELS if c[2] < ds]
    channels += [c for c in _TARGET_CHANNELS if c[2] < ds]
    channels += [c for c in _ACTION_CHANNELS if c[2] < da]
    return channels


def get_traj_format_text(ds, da):
    names = {n for n, _, _ in get_text_channels(ds, da)}
    lines = ['Each trajectory is provided as dimension-specific sequences '
             'over its time steps:']
    lines += [doc for key, doc in _CHANNEL_DOC if key in names]
    return '\n'.join(lines)


def _fmt_seq(vals, prec=3):
    return '[' + ', '.join(f'{v:.{prec}f}' for v in vals) + ']'


def text_projection(sa, ds, prec=3):
    da = sa.shape[1] - ds
    obs, act = sa[:, :ds], sa[:, ds:]
    lines = [f'num_steps: {sa.shape[0]}']
    for name, src, i in get_text_channels(ds, da):
        vals = obs[:, i] if src == 'obs' else act[:, i]
        lines.append(f'{name}: {_fmt_seq(vals, prec)}')
    return '\n'.join(lines)


def parse_text_trajectory(text, template_sa, ds):
    T = template_sa.shape[0]
    da = template_sa.shape[1] - ds
    out = template_sa.copy()
    found = 0
    for name, src, i in get_text_channels(ds, da):
        m = re.search(r'^\s*%s\s*:\s*\[([^\]]*)\]' % re.escape(name),
                      text, re.MULTILINE)
        if not m:
            continue
        try:
            vals = np.array([float(x) for x in m.group(1).split(',')
                             if x.strip() != ''], dtype=np.float32)
        except ValueError:
            continue
        if len(vals) != T:
            if len(vals) < max(3, T // 2):
                continue
            vals = np.concatenate([vals[:T],
                                   np.full(max(0, T - len(vals)),
                                           vals[-1], dtype=np.float32)])
        col = i if src == 'obs' else ds + i
        out[:, col] = vals
        found += 1
    if found < 4:
        return None
    if ds >= 36:
        out[1:, 18:36] = out[:-1, 0:18]
    return out


def extract_keyframes(sa, delta_v=0.005, delta_e=0.01,
                      smooth_topk=5, smooth_k=2, cpd_penalty=20,
                      max_keyframes=20):
    T = sa.shape[0]
    x = np.asarray(sa, dtype=np.float64)

    keys = {0, T - 1}

    v = np.linalg.norm(np.diff(x, axis=0), axis=1)
    for t in np.where(v < delta_v)[0]:
        keys.add(int(t + 1))

    win = 2 * smooth_k + 1
    if T >= win:
        smooth = uniform_filter1d(x, size=win, axis=0, mode='nearest')
        resid = np.linalg.norm(x - smooth, axis=1)
        cnt = 0
        for c in np.argsort(-resid):
            if cnt >= smooth_topk or resid[c] <= delta_e:
                break
            keys.add(int(c))
            cnt += 1

    try:
        import ruptures as rpt
        algo = rpt.Pelt(model='l2', min_size=3, jump=1).fit(x)
        for cp in algo.predict(pen=cpd_penalty):
            if 0 < cp < T:
                keys.add(int(cp))
    except Exception:
        pass

    keys = np.array(sorted(keys), dtype=np.int64)
    if len(keys) > max_keyframes:
        inner = keys[1:-1]
        take = np.linspace(0, len(inner) - 1, max_keyframes - 2).astype(int)
        keys = np.concatenate([[keys[0]], inner[take], [keys[-1]]])
        keys = np.unique(keys)
    return keys
