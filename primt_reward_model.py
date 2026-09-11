import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from reward_model import RewardModel
from primt_fm import (LLM_EVAL_SYSTEM, VLM_EVAL_SYSTEM, LLM_EVAL_USER,
                      VLM_EVAL_USER, HINDSIGHT_SYSTEM,
                      HINDSIGHT_ABDUCTION_USER, HINDSIGHT_ACTION_USER,
                      parse_preference, parse_causal_steps,
                      parse_counterfactual)
from primt_projection import (text_projection, parse_text_trajectory,
                              extract_keyframes, get_task_description,
                              get_traj_format_text, get_keyframe_params)
from primt_render import ENV_STATE_DIM, image_to_data_url, sliced_wasserstein

device = 'cuda'


class RunningScale(object):

    def __init__(self, fixed=None, momentum=0.05, eps=1e-8):
        self.fixed = float(fixed) if fixed else None
        self.momentum = momentum
        self.eps = eps
        self.mean = None

    def __call__(self, d):
        d = np.asarray(d, dtype=np.float64)
        if self.fixed is not None:
            return d / (self.fixed + self.eps)
        m = float(np.mean(np.abs(d)))
        if self.mean is None:
            self.mean = m
        elif m > 0:
            self.mean = (1 - self.momentum) * self.mean + self.momentum * m
        return d / ((self.mean or 0.0) + self.eps)


class PRIMTRewardModel(RewardModel):
    def __init__(self, ds, da,
                 fm=None,
                 renderer=None,
                 clip_embedder=None,
                 env_name='metaworld_button-press-v2',
                 seed=0,
                 crowd_k=3,
                 conf_alpha=0.5,
                 psl_w=(2.0, 1.0, 0.5),
                 intra_enable=True,
                 inter_mode='psl',
                 max_keyframes=8,
                 kf_params=None,
                 vd_scale=None,
                 td_scale=None,
                 fm_workers=4,
                 cf_enable=True,
                 cf_conf_thresh=0.8,
                 cf_verify=True,
                 cf_to_buffer=True,
                 cf_l1_thresh=0.3,
                 cf_margin=0.0,
                 cf_capacity=5000,
                 cf_batch=32,
                 cf_max_per_traj=5,
                 lambda_cf=0.05,
                 lambda_cf_adaptive=False,
                 fm_beta=12.0,
                 fm_noise=0.05,
                 cf_edit_width=4,
                 cf_smooth=2,
                 cf_offset=(0.05, 0.10),
                 **kwargs):
        super().__init__(ds, da, **kwargs)
        assert fm is not None, 'PRIMTRewardModel requires an FMClient'
        self.fm = fm
        self.renderer = renderer
        self.clip_embedder = clip_embedder
        self.task_description = get_task_description(env_name)
        self.dsa = ds + da
        self.format_text = get_traj_format_text(ds, da)
        self.intra_enable = intra_enable
        self.crowd_k = crowd_k if intra_enable else 1
        self.conf_alpha = conf_alpha
        self.inter_mode = inter_mode
        self.psl_w = psl_w
        self.max_keyframes = max_keyframes
        self.kf_params = kf_params or get_keyframe_params(env_name)
        self.vd_scale = RunningScale(vd_scale)
        self.td_scale = RunningScale(td_scale)
        self.fm_workers = fm_workers
        self.cf_enable = cf_enable
        self.cf_conf_thresh = cf_conf_thresh
        self.cf_verify = cf_verify
        self.cf_to_buffer = cf_to_buffer
        self.cf_l1_thresh = cf_l1_thresh
        self.cf_margin = cf_margin
        self.cf_capacity = cf_capacity
        self.cf_batch = cf_batch
        self.cf_max_per_traj = cf_max_per_traj
        self.lambda_cf = lambda_cf
        self.lambda_cf_adaptive = lambda_cf_adaptive
        self.scripted = getattr(fm, 'backend', '') == 'scripted'
        self.fm_beta = fm_beta
        self.fm_noise = fm_noise
        self.cf_edit_width = cf_edit_width
        self.cf_smooth = cf_smooth
        self.cf_offset = cf_offset
        self.rng = np.random.RandomState(seed + 12345)
        self._fail_lock = threading.Lock()

        self.cf_pref = np.empty((cf_capacity, self.size_segment, ds + da),
                                dtype=np.float32)
        self.cf_cf = np.empty((cf_capacity, self.size_segment, ds + da),
                              dtype=np.float32)
        self.cf_mask = np.empty((cf_capacity, self.size_segment),
                                dtype=np.float32)
        self.cf_index = 0
        self.cf_full = False

        self.last_label_acc = 0.0
        self.last_indecision_rate = 0.0
        self.last_llm_acc = 0.0
        self.last_vlm_acc = 0.0
        self.last_llm_indecision_rate = 0.0
        self.last_vlm_indecision_rate = 0.0
        self.last_cf_kept = 0
        self.last_cf_rejected = 0
        self.n_fm_failed = 0

    def r_hat_member(self, x, member=-1):
        if not torch.is_tensor(x):
            x = np.asarray(x)
            if x.shape[-1] == self.dsa + ENV_STATE_DIM:
                x = x[..., :self.dsa]
        return super().r_hat_member(x, member=member)

    def add_data(self, obs, act, rew, done, env_state=None):
        if env_state is None:
            env_state = np.zeros(ENV_STATE_DIM, dtype=np.float32)
        sa_t = np.concatenate([obs, act, env_state], axis=-1)
        r_t = rew
        flat_input = sa_t.reshape(1, self.dsa + ENV_STATE_DIM)
        flat_target = np.array(r_t).reshape(1, 1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate(
                    [self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate(
                    [self.targets[-1], flat_target])


    def _fuse_votes(self, votes, confs):
        votes = np.asarray(votes)
        confs = np.asarray(confs, dtype=np.float32)
        if not self.intra_enable:
            return int(votes[0]), float(confs[0])
        cand, cnt = np.unique(votes, return_counts=True)
        winners = cand[cnt == cnt.max()]
        lab = int(winners[0]) if len(winners) == 1 else -1
        agree = votes == lab
        avg_c = float(confs[agree].mean()) if agree.any() else 0.0
        cons = float(agree.mean())
        conf = self.conf_alpha * avg_c + (1 - self.conf_alpha) * cons
        return lab, conf

    @staticmethod
    def _unswap(label, swapped):
        if label is None:
            return -1
        if swapped and label in (0, 1):
            return 1 - label
        return label

    def _note_failure(self):
        with self._fail_lock:
            self.n_fm_failed += 1

    def _modality_scores(self, r_t, modality):
        r = r_t[..., 0]
        if modality == 'vlm':
            w = np.linspace(0.5, 1.5, r.shape[1])[None, :]
            return (r * w).sum(axis=1)
        progress = r[:, -5:].mean(axis=1) - r[:, :5].mean(axis=1)
        return r.sum(axis=1) + progress * r.shape[1] * 0.1

    def _scripted_intra(self, r_t_1, r_t_2, modality):
        s1 = self._modality_scores(r_t_1, modality)
        s2 = self._modality_scores(r_t_2, modality)
        scale = np.std(np.concatenate([s1, s2])) + 1e-6
        B = s1.shape[0]
        labs = np.zeros(B, dtype=np.int64)
        confs = np.zeros(B, dtype=np.float32)
        for b in range(B):
            votes, vconf = [], []
            for _ in range(self.crowd_k):
                d = (s1[b] - s2[b]) / scale + self.rng.randn() * self.fm_noise
                p1 = 1.0 / (1.0 + np.exp(-self.fm_beta * d))
                c = abs(p1 - 0.5) * 2.0
                lab = 0 if p1 > 0.5 else 1
                if c < 0.05:
                    lab = -1
                votes.append(lab)
                vconf.append(c)
            labs[b], confs[b] = self._fuse_votes(votes, vconf)
        return labs, confs

    def _llm_votes(self, text1, text2, swaps, salt_prefix):
        votes, confs, reasoning = [], [], ''
        for k in range(self.crowd_k):
            t1, t2 = (text2, text1) if swaps[k] else (text1, text2)
            prompt = LLM_EVAL_USER.format(
                task_description=self.task_description,
                n=self.size_segment, format_text=self.format_text,
                text1=t1, text2=t2)
            try:
                resp = self.fm.chat(LLM_EVAL_SYSTEM, prompt,
                                    model=self.fm.llm_model,
                                    salt=f'{salt_prefix}_llm_k{k}',
                                    kind='llm_eval')
                lab, conf, reas = parse_preference(resp)
            except Exception as e:
                print(f'[PRIMT] LLM vote failed: {e}')
                self._note_failure()
                lab, conf, reas = None, 0.0, ''
            votes.append(self._unswap(lab, swaps[k]))
            confs.append(conf)
            if reas and not reasoning:
                reasoning = reas
        lab, conf = self._fuse_votes(votes, confs)
        return lab, conf, reasoning

    def _vlm_votes(self, urls1, keys1, urls2, keys2, swaps, salt_prefix):
        votes, confs, reasoning = [], [], ''
        for k in range(self.crowd_k):
            if swaps[k]:
                ua, ka, ub, kb = urls2, keys2, urls1, keys1
            else:
                ua, ka, ub, kb = urls1, keys1, urls2, keys2
            prompt = VLM_EVAL_USER.format(
                task_description=self.task_description)
            content = [{'type': 'text', 'text': prompt},
                       {'type': 'text',
                        'text': f'Trajectory 1 keyframes '
                                f'(step indices {list(ka)}):'}]
            content += [{'type': 'image_url', 'image_url': {'url': u}}
                        for u in ua]
            content += [{'type': 'text',
                         'text': f'Trajectory 2 keyframes '
                                 f'(step indices {list(kb)}):'}]
            content += [{'type': 'image_url', 'image_url': {'url': u}}
                        for u in ub]
            try:
                resp = self.fm.chat(VLM_EVAL_SYSTEM, content,
                                    model=self.fm.vlm_model,
                                    salt=f'{salt_prefix}_vlm_k{k}',
                                    kind='vlm_eval')
                lab, conf, reas = parse_preference(resp)
            except Exception as e:
                print(f'[PRIMT] VLM vote failed: {e}')
                self._note_failure()
                lab, conf, reas = None, 0.0, ''
            votes.append(self._unswap(lab, swaps[k]))
            confs.append(conf)
            if reas and not reasoning:
                reasoning = reas
        lab, conf = self._fuse_votes(votes, confs)
        return lab, conf, reasoning


    def _td_high(self, sa_1, sa_2):
        def trjvol(x):
            dd = x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :]
            return np.linalg.norm(dd, axis=-1).mean(axis=1)
        diff = np.abs(trjvol(sa_1) - trjvol(sa_2))
        return 1.0 / (1.0 + np.exp(-self.td_scale(diff)))

    def _vd_high(self, embs_1, embs_2):
        d = np.array([sliced_wasserstein(e1, e2)
                      for e1, e2 in zip(embs_1, embs_2)], dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-self.vd_scale(d)))

    def _vd_high_state(self, sa_1, sa_2):
        from scipy.stats import wasserstein_distance
        B = sa_1.shape[0]
        d = np.zeros(B, dtype=np.float32)
        dims = np.arange(0, self.ds, max(1, self.ds // 12))
        for b in range(B):
            d[b] = np.mean([wasserstein_distance(sa_1[b, :, i], sa_2[b, :, i])
                            for i in dims])
        return 1.0 / (1.0 + np.exp(-self.vd_scale(d)))


    def _inter_fuse(self, lab_v, conf_v, lab_l, conf_l, vd, td):
        if self.inter_mode == 'maxconf':
            labels = np.array(
                [lab_v[b] if conf_v[b] >= conf_l[b] else lab_l[b]
                 for b in range(len(lab_v))], dtype=np.int64)
            score = np.maximum(np.asarray(conf_v, dtype=np.float32),
                               np.asarray(conf_l, dtype=np.float32))
            return labels, score
        return self._psl_fuse(lab_v, conf_v, lab_l, conf_l, vd, td)

    def _psl_fuse(self, lab_v, conf_v, lab_l, conf_l, vd, td):
        B = len(lab_v)
        labels = np.zeros(B, dtype=np.int64)
        psl_score = np.zeros(B, dtype=np.float32)
        LAB = (-1, 0, 1)
        idx_of = {-1: 0, 0: 1, 1: 2}
        for b in range(B):
            cv, cl = float(conf_v[b]), float(conf_l[b])
            vdb, tdb = float(vd[b]), float(td[b])
            terms = []
            for l in LAB:
                j = idx_of[l]
                is_agree = 1.0 if (lab_v[b] == l and lab_l[b] == l) else 0.0
                vlm_lab = 1.0 if lab_v[b] == l else 0.0
                llm_lab = 1.0 if lab_l[b] == l else 0.0
                for confM in (cv, cl):
                    body = is_agree + confM
                    terms.append((self.psl_w[0], j, body - 1.0))
                body = (1.0 - is_agree) + vlm_lab + cv + vdb
                terms.append((self.psl_w[1], j, body - 3.0))
                body = (1.0 - is_agree) + llm_lab + cl + tdb
                terms.append((self.psl_w[1], j, body - 3.0))
            body = (1.0 - cv) + (1.0 - cl)
            terms.append((self.psl_w[2], idx_of[-1], body - 1.0))

            lab, score = self._psl_solve(terms)
            labels[b] = LAB[lab]
            psl_score[b] = score
        return labels, psl_score

    def _psl_solve(self, terms):
        from scipy.optimize import minimize

        def obj(Y):
            f = 0.0
            g = np.zeros(3)
            for w, j, c in terms:
                d = c - Y[j]
                if d > 0.0:
                    f += w * d * d
                    g[j] += -2.0 * w * d
            return f, g

        cons = ({'type': 'eq', 'fun': lambda Y: np.sum(Y) - 1.0,
                 'jac': lambda Y: np.ones(3)},)
        bounds = [(0.0, 1.0)] * 3
        y0 = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        try:
            res = minimize(obj, y0, jac=True, bounds=bounds,
                           constraints=cons, method='SLSQP',
                           options={'maxiter': 100, 'ftol': 1e-8})
            Y = np.clip(res.x, 0.0, 1.0)
            s = Y.sum()
            Y = Y / s if s > 1e-8 else np.array([1.0, 0.0, 0.0])
        except Exception:
            Y = np.array([1.0, 0.0, 0.0])
        j = int(np.argmax(Y))
        return j, float(Y[j])


    def _log_diagnostics(self, labels, lab_l, lab_v, r_t_1, r_t_2):
        gt = 1 * (r_t_1.sum(axis=1) < r_t_2.sum(axis=1))
        gt_flat = gt.flatten()
        decided = (labels != -1).flatten()
        if decided.any():
            self.last_label_acc = float(
                (labels[decided].flatten() == gt[decided].flatten()).mean())
        self.last_indecision_rate = float((~decided).mean())
        lab_l_arr = np.asarray(lab_l).flatten()
        lab_v_arr = np.asarray(lab_v).flatten()
        dec_l = lab_l_arr != -1
        dec_v = lab_v_arr != -1
        if dec_l.any():
            self.last_llm_acc = float(
                (lab_l_arr[dec_l] == gt_flat[dec_l]).mean())
        self.last_llm_indecision_rate = float((~dec_l).mean())
        if dec_v.any():
            self.last_vlm_acc = float(
                (lab_v_arr[dec_v] == gt_flat[dec_v]).mean())
        self.last_vlm_indecision_rate = float((~dec_v).mean())

    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        B = sa_t_1.shape[0]
        sa1, st1 = sa_t_1[..., :self.dsa], sa_t_1[..., self.dsa:]
        sa2, st2 = sa_t_2[..., :self.dsa], sa_t_2[..., self.dsa:]

        if self.scripted:
            return self._get_label_scripted(sa1, sa2, r_t_1, r_t_2)

        texts1 = [text_projection(sa1[b], self.ds) for b in range(B)]
        texts2 = [text_projection(sa2[b], self.ds) for b in range(B)]
        keys1 = [extract_keyframes(sa1[b], max_keyframes=self.max_keyframes,
                                   **self.kf_params) for b in range(B)]
        keys2 = [extract_keyframes(sa2[b], max_keyframes=self.max_keyframes,
                                   **self.kf_params) for b in range(B)]
        imgs1 = [self.renderer.render_flat_states(st1[b][keys1[b]])
                 for b in range(B)]
        imgs2 = [self.renderer.render_flat_states(st2[b][keys2[b]])
                 for b in range(B)]
        urls1 = [[image_to_data_url(im) for im in ims] for ims in imgs1]
        urls2 = [[image_to_data_url(im) for im in ims] for ims in imgs2]
        embs1 = [self.clip_embedder.embed(ims) for ims in imgs1]
        embs2 = [self.clip_embedder.embed(ims) for ims in imgs2]

        swaps_l = self.rng.rand(B, self.crowd_k) < 0.5
        swaps_v = self.rng.rand(B, self.crowd_k) < 0.5
        cf_swaps = self.rng.rand(B, self.cf_max_per_traj, self.crowd_k) < 0.5
        salt_base = self.rng.randint(1 << 30)

        def eval_pair(b):
            sp = f'q{salt_base}_{b}'
            ll = self._llm_votes(texts1[b], texts2[b], swaps_l[b], sp)
            vv = self._vlm_votes(urls1[b], keys1[b], urls2[b], keys2[b],
                                 swaps_v[b], sp)
            return ll, vv

        with ThreadPoolExecutor(max_workers=self.fm_workers) as ex:
            results = list(ex.map(eval_pair, range(B)))
        lab_l = [r[0][0] for r in results]
        conf_l = [r[0][1] for r in results]
        reas_l = [r[0][2] for r in results]
        lab_v = [r[1][0] for r in results]
        conf_v = [r[1][1] for r in results]
        reas_v = [r[1][2] for r in results]

        td = self._td_high(sa1, sa2)
        vd = self._vd_high(embs1, embs2)
        fused_lab, _ = self._inter_fuse(lab_v, conf_v, lab_l, conf_l, vd, td)
        labels = fused_lab.reshape(-1, 1)

        self._log_diagnostics(labels, lab_l, lab_v, r_t_1, r_t_2)

        if self.cf_enable:
            gate = np.maximum(np.asarray(conf_v, dtype=np.float32),
                              np.asarray(conf_l, dtype=np.float32))
            self._hindsight_augment(sa1, sa2, texts1, texts2, keys1, keys2,
                                    reas_l, reas_v, fused_lab, gate,
                                    cf_swaps, salt_base)
        return sa1, sa2, r_t_1, r_t_2, labels

    def _get_label_scripted(self, sa1, sa2, r_t_1, r_t_2):
        lab_l, conf_l = self._scripted_intra(r_t_1, r_t_2, 'llm')
        lab_v, conf_v = self._scripted_intra(r_t_1, r_t_2, 'vlm')
        td = self._td_high(sa1, sa2)
        vd = self._vd_high_state(sa1, sa2)
        fused_lab, _ = self._inter_fuse(lab_v, conf_v, lab_l, conf_l, vd, td)
        labels = fused_lab.reshape(-1, 1)

        self._log_diagnostics(labels, lab_l, lab_v, r_t_1, r_t_2)

        if self.cf_enable:
            gate = np.maximum(conf_v, conf_l)
            self._hindsight_scripted(sa1, sa2, r_t_1, r_t_2, fused_lab, gate)
        return sa1, sa2, r_t_1, r_t_2, labels


    def _edit_l1(self, cf, ref, mask):
        edited = mask > 0
        if not edited.any():
            return np.inf
        num = np.abs(cf[edited] - ref[edited]).sum()
        den = np.abs(ref[edited]).sum() + 1e-6
        return float(num / den)

    def _scripted_progress(self, sa):
        ee, obj = sa[:, 0:3], sa[:, 4:7]
        return float(-np.linalg.norm(ee - obj, axis=-1).mean())

    def _scripted_verify(self, pref_sa, cf_sa):
        s_p = self._scripted_progress(pref_sa)
        s_c = self._scripted_progress(cf_sa)
        scale = abs(s_p) + abs(s_c) + 1e-6
        votes, confs = [], []
        for _ in range(self.crowd_k):
            d = (s_p - s_c) / scale + self.rng.randn() * self.fm_noise
            p = 1.0 / (1.0 + np.exp(-self.fm_beta * d))
            c = abs(p - 0.5) * 2.0
            lab = 0 if p > 0.5 else 1
            if c < 0.05:
                lab = -1
            votes.append(lab)
            confs.append(c)
        lab, conf = self._fuse_votes(votes, confs)
        return lab == 0

    def _hindsight_scripted(self, sa1, sa2, r_t_1, r_t_2, labels, gate):
        kept, rejected = 0, 0
        for b in range(len(labels)):
            if labels[b] not in (0, 1) or gate[b] < self.cf_conf_thresh:
                continue
            pref_sa = sa1[b] if labels[b] == 0 else sa2[b]
            pref_r = r_t_1[b] if labels[b] == 0 else r_t_2[b]
            for _ in range(self.cf_max_per_traj):
                out = self._make_counterfactual(pref_sa, pref_r)
                if out is None:
                    rejected += 1
                    continue
                if self.cf_verify and not self._scripted_verify(pref_sa,
                                                                out[0]):
                    rejected += 1
                    continue
                self._cf_put(pref_sa, out[0], out[1])
                kept += 1
        self.last_cf_kept = kept
        self.last_cf_rejected = rejected

    def _make_counterfactual(self, sa, r):
        T = sa.shape[0]
        dr = np.abs(np.diff(r[:, 0]))
        valid = np.arange(self.cf_smooth + 1,
                          T - self.cf_edit_width - self.cf_smooth - 1)
        if len(valid) == 0:
            return None
        cand = valid[np.argsort(-dr[valid - 1])[:5]]
        t_star = int(self.rng.choice(cand))
        cf = sa.copy()
        mask = np.zeros(T, dtype=np.float32)
        offset = self.rng.uniform(*self.cf_offset)
        for t in range(t_star, t_star + self.cf_edit_width):
            cf[t, self.ds:] = 0.0
            ee, obj = sa[t, 0:3], sa[t, 4:7]
            d = ee - obj
            n = np.linalg.norm(d)
            direction = d / n if n > 1e-6 else self.rng.randn(3) / np.sqrt(3.0)
            cf[t, 0:3] = ee + direction * offset
            mask[t] = 1.0
        for k in range(1, self.cf_smooth + 1):
            a = k / float(self.cf_smooth + 1)
            tl, tr = t_star - k, t_star + self.cf_edit_width - 1 + k
            cf[tl] = a * sa[tl] + (1 - a) * cf[t_star]
            cf[tr] = a * sa[tr] + (1 - a) * cf[t_star + self.cf_edit_width - 1]
            mask[tl] = 1.0
            mask[tr] = 1.0
        if self._edit_l1(cf, sa, mask) > self.cf_l1_thresh:
            return None
        return cf, mask

    def _hindsight_augment(self, sa1, sa2, texts1, texts2, keys1, keys2,
                           reas_l, reas_v, labels, gate, cf_swaps, salt_base):
        idx = [b for b in range(len(labels))
               if labels[b] in (0, 1) and gate[b] >= self.cf_conf_thresh]
        if not idx:
            self.last_cf_kept = 0
            self.last_cf_rejected = 0
            return

        def make_cfs(b):
            pref_is_1 = labels[b] == 0
            pref_sa = sa1[b] if pref_is_1 else sa2[b]
            pref_text = texts1[b] if pref_is_1 else texts2[b]
            sp = f'h{salt_base}_{b}'
            sys = HINDSIGHT_SYSTEM
            u1 = HINDSIGHT_ABDUCTION_USER.format(
                task_description=self.task_description,
                format_text=self.format_text,
                text1=texts1[b], text2=texts2[b],
                trj_preferred='1' if pref_is_1 else '2',
                kvist1=list(keys1[b]), kvist2=list(keys2[b]),
                llm_reasoning=reas_l[b] or 'n/a',
                vlm_reasoning=reas_v[b] or 'n/a')
            r1 = self.fm.chat(sys, u1, salt=sp + '_abd', kind='hs_abduction')
            causal_steps, _ = parse_causal_steps(r1)
            if not causal_steps:
                return [], self.cf_max_per_traj

            ref_sa = parse_text_trajectory(pref_text, pref_sa, self.ds)
            if ref_sa is None:
                ref_sa = pref_sa
            messages = [{'role': 'system', 'content': sys},
                        {'role': 'user', 'content': u1},
                        {'role': 'assistant', 'content': r1},
                        {'role': 'user', 'content': HINDSIGHT_ACTION_USER}]

            kept, rejected = [], 0
            for j in range(self.cf_max_per_traj):
                r2 = self.fm.chat_messages(messages, salt=f'{sp}_act{j}',
                                           kind='hs_action', max_tokens=8192)
                _, _, traj_text = parse_counterfactual(r2)
                if not traj_text:
                    rejected += 1
                    continue
                cf_sa = parse_text_trajectory(traj_text, pref_sa, self.ds)
                if cf_sa is None:
                    rejected += 1
                    continue
                diff = np.abs(cf_sa - ref_sa).max(axis=1)
                mask = (diff > 2e-3).astype(np.float32)
                if mask.sum() == 0 or mask.sum() > 0.5 * len(mask):
                    rejected += 1
                    continue
                if self._edit_l1(cf_sa, ref_sa, mask) > self.cf_l1_thresh:
                    rejected += 1
                    continue
                if self.cf_verify:
                    cf_text = text_projection(cf_sa, self.ds)
                    vlab, _, _ = self._llm_votes(pref_text, cf_text,
                                                 cf_swaps[b][j],
                                                 f'{sp}_ver{j}')
                    if vlab != 0:
                        rejected += 1
                        continue
                kept.append((pref_sa, cf_sa, mask))
            return kept, rejected

        def safe_make_cfs(b):
            try:
                return make_cfs(b)
            except Exception as e:
                print(f'[PRIMT] hindsight cf failed for pair {b}: {e}')
                self._note_failure()
                return [], self.cf_max_per_traj

        with ThreadPoolExecutor(max_workers=self.fm_workers) as ex:
            outs = list(ex.map(safe_make_cfs, idx))
        kept, rejected = 0, 0
        for cfs, n_rej in outs:
            rejected += n_rej
            for out in cfs:
                self._cf_put(*out)
                kept += 1
        self.last_cf_kept = kept
        self.last_cf_rejected = rejected

    def _cf_put(self, pref, cf, mask):
        i = self.cf_index
        self.cf_pref[i] = pref
        self.cf_cf[i] = cf
        self.cf_mask[i] = mask
        self.cf_index = (i + 1) % self.cf_capacity
        if self.cf_index == 0:
            self.cf_full = True
        if self.cf_to_buffer:
            self.put_queries(
                np.asarray(pref, dtype=np.float32)[None],
                np.asarray(cf, dtype=np.float32)[None],
                np.zeros((1, 1), dtype=np.float32))

    def _cf_len(self):
        return self.cf_capacity if self.cf_full else self.cf_index


    def _causal_aux_loss(self, member):
        n = self._cf_len()
        if n == 0:
            return None
        idx = self.rng.choice(n, size=min(self.cf_batch, n), replace=False)
        pref = self.cf_pref[idx]
        cf = self.cf_cf[idx]
        mask = torch.from_numpy(self.cf_mask[idx]).float().to(device)
        r_p = self.r_hat_member(pref, member=member).squeeze(-1)
        r_c = self.r_hat_member(cf, member=member).squeeze(-1)
        contrast = mask * F.softplus(self.cf_margin + r_c - r_p)
        consist = (1 - mask) * (r_p - r_c) ** 2
        loss = (contrast.sum(dim=1) + consist.sum(dim=1)).mean()
        return loss

    def train_reward_primt(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = [np.random.permutation(max_len)
                             for _ in range(self.de)]
        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            last_index = min((epoch + 1) * self.train_batch_size, max_len)
            for member in range(self.de):
                idxs = total_batch_index[member][
                    epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(
                    labels.flatten()).long().to(device)
                if member == 0:
                    total += labels.size(0)
                r_hat1 = self.r_hat_member(sa_t_1, member=member).sum(axis=1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member).sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                uniform_index = labels == -1
                labels_c = labels.clone()
                labels_c[uniform_index] = 0
                target = torch.zeros_like(r_hat).scatter(
                    1, labels_c.unsqueeze(1), self.label_target)
                target += self.label_margin
                if uniform_index.sum() > 0:
                    target[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target)

                aux = self._causal_aux_loss(member)
                if aux is not None:
                    scale = 1.0
                    if self.lambda_cf_adaptive:
                        scale = (curr_loss.detach()
                                 / (aux.detach().abs() + 1e-8))
                    curr_loss = curr_loss + self.lambda_cf * scale * aux
                loss += curr_loss

                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels_c).sum().item()
                ensemble_acc[member] += correct
            loss.backward()
            self.opt.step()
        return ensemble_acc / total
