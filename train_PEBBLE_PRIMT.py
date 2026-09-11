import numpy as np
import torch
import os
import time

from logger import Logger
from replay_buffer import ReplayBuffer
from primt_fm import FMClient
from primt_render import TrajectoryRenderer, CLIPEmbedder, pack_env_state
from primt_projection import get_keyframe_params
from primt_reward_model import PRIMTRewardModel
from primt_foresight import (inject_foresight_into_reward_model,
                             inject_foresight_scripted)
from collections import deque
import pandas as pd

import utils
import hydra


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent='sac')

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False

        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        self.inner_env = self.env
        while hasattr(self.inner_env, 'env'):
            self.inner_env = self.inner_env.env

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)

        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        self.fm = FMClient(
            backend=cfg.primt.fm.backend,
            llm_model=cfg.primt.fm.llm_model,
            vlm_model=cfg.primt.fm.vlm_model,
            api_key_env=cfg.primt.fm.api_key_env,
            base_url=cfg.primt.fm.base_url or None,
            temperature=cfg.primt.fm.temperature,
            max_retries=cfg.primt.fm.max_retries,
            cache_dir=os.path.join(self.work_dir, 'fm_cache'),
            log_path=os.path.join(self.work_dir, 'fm_calls.jsonl'),
            seed=cfg.seed)
        self.scripted = (cfg.primt.fm.backend == 'scripted')
        if self.scripted:
            print('[PRIMT] backend=scripted: preference labels and hindsight '
                  'counterfactuals are produced by a GT-proxy simulation of '
                  'the LLM/VLM evaluators, which reads the environment reward. '
                  'This is a cost-free stand-in for the foundation-model path, '
                  'NOT the PRIMT method itself -- results from this backend '
                  'are an oracle-labelled reference, not the paper result. '
                  'Use primt.fm.backend=openai to run PRIMT.')
            self.renderer = None
            self.clip_embedder = None
        else:
            self.renderer = TrajectoryRenderer(
                cfg.env, camera_name=cfg.primt.fm.camera,
                resolution=cfg.primt.fm.resolution)
            self.clip_embedder = CLIPEmbedder(device=cfg.device)

        kf_override = {k: cfg.primt.keyframe[k] for k in
                       ('delta_v', 'delta_e', 'smooth_k', 'smooth_topk',
                        'cpd_penalty')}

        self.reward_model = PRIMTRewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal,
            max_size=cfg.primt.traj_buffer_size,
            fm=self.fm,
            renderer=self.renderer,
            clip_embedder=self.clip_embedder,
            env_name=cfg.env,
            seed=cfg.seed,
            crowd_k=cfg.primt.crowd_k,
            conf_alpha=cfg.primt.conf_alpha,
            intra_enable=cfg.primt.intra_enable,
            inter_mode=cfg.primt.inter_mode,
            max_keyframes=cfg.primt.max_keyframes,
            kf_params=get_keyframe_params(cfg.env, kf_override),
            vd_scale=cfg.primt.vd_scale,
            td_scale=cfg.primt.td_scale,
            fm_workers=cfg.primt.fm.workers,
            cf_enable=cfg.primt.cf_enable,
            cf_conf_thresh=cfg.primt.cf_conf_thresh,
            cf_verify=cfg.primt.cf_verify,
            cf_to_buffer=cfg.primt.cf_to_buffer,
            cf_l1_thresh=cfg.primt.cf_l1_thresh,
            cf_batch=cfg.primt.cf_batch,
            cf_max_per_traj=cfg.primt.cf_max_per_traj,
            lambda_cf=cfg.primt.lambda_cf,
            lambda_cf_adaptive=cfg.primt.lambda_cf_adaptive)

    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0

        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)

                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success

        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0

        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward',
                        average_true_episode_reward, self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate, self.step)
            self.logger.log('train/true_episode_success', success_rate,
                            self.step)
        self.logger.dump(self.step)

    def learn_reward(self, first_flag=0):
        labeled_queries = 0
        if first_flag == 1:
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError

        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        train_acc = 0
        total_acc = 0
        if self.labeled_feedback > 0:
            for epoch in range(self.cfg.reward_update):
                train_acc = self.reward_model.train_reward_primt()
                total_acc = np.mean(train_acc)
                if total_acc > 0.97:
                    break

        print(("Reward updated! ACC: {:.3f} | fused label acc: {:.3f} | "
               "LLM acc: {:.3f} | VLM acc: {:.3f} | indecision: {:.3f} | "
               "cf buffer: {} (+{} kept / {} rejected) | "
               "FM calls: {} (cache hits: {}, failed: {}) | "
               "tokens {:.0f}k in / {:.0f}k out").format(
            total_acc, self.reward_model.last_label_acc,
            self.reward_model.last_llm_acc, self.reward_model.last_vlm_acc,
            self.reward_model.last_indecision_rate,
            self.reward_model._cf_len(), self.reward_model.last_cf_kept,
            self.reward_model.last_cf_rejected,
            self.fm.n_calls, self.fm.n_cache_hits,
            self.reward_model.n_fm_failed,
            self.fm.tokens_in / 1e3, self.fm.tokens_out / 1e3))

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0

        if self.cfg.primt.num_foresight_traj > 0:
            if self.scripted:
                n = inject_foresight_scripted(
                    self.reward_model, self.cfg.env,
                    self.cfg.primt.num_foresight_traj, self.cfg.seed)
            else:
                n = inject_foresight_into_reward_model(
                    self.reward_model, self.fm, self.cfg.env,
                    self.cfg.primt.num_foresight_traj, self.cfg.seed)
            print(f'[PRIMT] injected {n} foresight trajectories '
                  f'(FM calls so far: {self.fm.n_calls})')

        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()

        interact_count = 0
        while self.step < self.cfg.num_train_steps:

            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)
                self.logger.log('train/true_episode_reward',
                                true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback,
                                self.step)
                self.logger.log('train/labeled_feedback',
                                self.labeled_feedback, self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                                    self.step)
                    self.logger.log('train/true_episode_success',
                                    episode_success, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            if self.step == (self.cfg.num_seed_steps +
                             self.cfg.num_unsup_steps):
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps - self.step) \
                        / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps \
                        / (self.cfg.num_train_steps - self.step + 1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)

                new_margin = np.mean(avg_train_true_return) \
                    * (self.cfg.segment / self.env._max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)

                self.learn_reward(first_flag=1)

                self.replay_buffer.relabel_with_predictor(self.reward_model)

                self.agent.reset_critic()

                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=self.cfg.reset_update,
                    policy_update=True)

                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps - self.step) \
                                / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps \
                                / (self.cfg.num_train_steps - self.step + 1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)

                        new_margin = np.mean(avg_train_true_return) \
                            * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(
                            new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(
                            new_margin * self.cfg.teacher_eps_equal)

                        if self.reward_model.mb_size + self.total_feedback \
                                > self.cfg.max_feedback:
                            self.reward_model.set_batch(
                                self.cfg.max_feedback - self.total_feedback)

                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(
                            self.reward_model)
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger,
                                  self.step, 1)

            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=1, K=self.cfg.topK)

            env_state = pack_env_state(self.env.get_env_state(),
                                       self.inner_env)
            next_obs, reward, done, extra = self.env.step(action)
            reward_hat = self.reward_model.r_hat(
                np.concatenate([obs, action], axis=-1))

            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps \
                else done
            episode_reward += reward_hat
            true_episode_reward += reward

            if self.log_success:
                episode_success = max(episode_success, extra['success'])

            self.reward_model.add_data(obs, action, reward, done,
                                       env_state=env_state)
            self.replay_buffer.add(
                obs, action, reward_hat,
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)


@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
