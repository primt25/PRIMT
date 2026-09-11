import io
import base64

import numpy as np

_NQ, _NV = 10, 10

ENV_STATE_DIM = 1 + _NQ + _NV + 3 + 4 + 6

_LAYOUT_OFF = 1 + _NQ + _NV + 3 + 4

_OBJ_BODY_NAMES = ('box', 'door', 'obj', 'button', 'drawer', 'peg')


def pack_env_state(state, env=None):
    sim_state, (mocap_pos, mocap_quat) = state
    qpos = np.asarray(sim_state.qpos, dtype=np.float64)
    qvel = np.asarray(sim_state.qvel, dtype=np.float64)
    if qpos.size != _NQ or qvel.size != _NV:
        raise ValueError(
            f'pack_env_state expects nq={_NQ}, nv={_NV} (the supported '
            f'MetaWorld tasks) but got nq={qpos.size}, nv={qvel.size}. '
            'Update _NQ/_NV/ENV_STATE_DIM in primt_render.py for this env.')

    layout = np.zeros(6, dtype=np.float64)
    if env is not None:
        obj = getattr(env, 'obj_init_pos', None)
        tgt = getattr(env, '_target_pos', None)
        if obj is not None:
            layout[0:3] = np.asarray(obj, dtype=np.float64).ravel()[:3]
        if tgt is not None:
            layout[3:6] = np.asarray(tgt, dtype=np.float64).ravel()[:3]

    return np.concatenate([
        [sim_state.time],
        qpos,
        qvel,
        np.asarray(mocap_pos, dtype=np.float64).ravel(),
        np.asarray(mocap_quat, dtype=np.float64).ravel(),
        layout,
    ]).astype(np.float32)


def unpack_env_state(flat):
    flat = np.asarray(flat, dtype=np.float64)
    o = _LAYOUT_OFF
    return (flat[0], flat[1:1 + _NQ], flat[1 + _NQ:1 + _NQ + _NV],
            flat[o - 7:o - 4].reshape(1, 3), flat[o - 4:o].reshape(1, 4),
            flat[o:o + 3], flat[o + 3:o + 6])


class TrajectoryRenderer(object):

    def __init__(self, env_name, camera_name='corner', resolution=224):
        self.env_name = env_name.replace('metaworld_', '')
        self.camera_name = camera_name
        self.resolution = resolution
        self._env = None
        self._obj_body_id = None
        self._obj_body_searched = False

    def _lazy_env(self):
        if self._env is None:
            import metaworld.envs.mujoco.env_dict as _env_dict
            env_cls = _env_dict.ALL_V2_ENVIRONMENTS[self.env_name]
            env = env_cls()
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.reset()
            self._env = env
        return self._env

    def _obj_body(self, env):
        if not self._obj_body_searched:
            self._obj_body_searched = True
            for name in _OBJ_BODY_NAMES:
                try:
                    self._obj_body_id = env.sim.model.body_name2id(name)
                    break
                except Exception:
                    continue
        return self._obj_body_id

    def _restore_layout(self, env, obj_pos, tgt_pos):
        if not np.any(obj_pos) and not np.any(tgt_pos):
            return
        if np.any(obj_pos):
            env.obj_init_pos = obj_pos.copy()
            body_id = self._obj_body(env)
            if body_id is not None:
                env.sim.model.body_pos[body_id] = obj_pos
        if np.any(tgt_pos):
            env._target_pos = tgt_pos.copy()
            try:
                env._set_pos_site('goal', tgt_pos)
            except Exception:
                pass

    def render_flat_states(self, flat_states):
        from mujoco_py import MjSimState
        env = self._lazy_env()
        frames = []
        for fs in np.atleast_2d(flat_states):
            t, qpos, qvel, mpos, mquat, obj_pos, tgt_pos = unpack_env_state(fs)
            self._restore_layout(env, obj_pos, tgt_pos)
            env.sim.set_state(MjSimState(t, qpos, qvel, None, {}))
            env.sim.data.mocap_pos[:] = mpos
            env.sim.data.mocap_quat[:] = mquat
            env.sim.forward()
            img = env.render(offscreen=True, camera_name=self.camera_name,
                             resolution=(self.resolution, self.resolution))
            frames.append(np.ascontiguousarray(img))
        return frames


def image_to_data_url(img, quality=85):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


class CLIPEmbedder(object):

    def __init__(self, device='cuda'):
        self.device = device
        self._model = None
        self._preprocess = None

    def _lazy(self):
        if self._model is None:
            import clip
            self._model, self._preprocess = clip.load('ViT-B/32',
                                                       device=self.device)
            self._model.eval()
        return self._model, self._preprocess

    def embed(self, images):
        import torch
        from PIL import Image
        model, preprocess = self._lazy()
        batch = torch.stack([preprocess(Image.fromarray(im))
                             for im in images]).to(self.device)
        with torch.no_grad():
            emb = model.encode_image(batch).float()
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb.cpu().numpy()


_SW_DIRS = None


def sliced_wasserstein(emb_a, emb_b, n_proj=64, seed=0):
    global _SW_DIRS
    d = emb_a.shape[1]
    if _SW_DIRS is None or _SW_DIRS.shape[1] != d:
        rng = np.random.RandomState(seed)
        dirs = rng.randn(n_proj, d)
        _SW_DIRS = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    q = np.linspace(0.0, 1.0, 32)
    pa = emb_a @ _SW_DIRS.T
    pb = emb_b @ _SW_DIRS.T
    qa = np.quantile(pa, q, axis=0)
    qb = np.quantile(pb, q, axis=0)
    return float(np.abs(qa - qb).mean())
