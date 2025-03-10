import numpy as np
import torch
import argparse
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy
import numpy

import utils
from utils import predDataset
from logger_bilayer import Logger
from video import VideoRecorder

from sac_ae_bilayer_ssa_bpp import SacAeAgent

import gc
import objgraph

from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

qz = [1000., 600., 1.]

# xvfb-run -a -s "-screen 0 640x480x24"


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cartpole')
    parser.add_argument('--task_name', default='swingup')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='sac_ae', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)#1000 0
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=100, type=int)#50
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--decoder_type', default='pixel', type=str)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=6, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')

    parser.add_argument('--testpsnr', default=False, action='store_true')
    parser.add_argument('--q', default=False, action='store_true')

    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step, testpsnr=False, q=False):
    z1mQ_list = []
    z2mQ_list = []
    z3mQ_list = []
    action_list = []
    nextobs_list = []
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        episode_psnr_z1 = []
        episode_psnr_z2 = []
        episode_psnr_z3 = []
        episode_bpp_z1 = []
        episode_bpp_z2 = []
        episode_bpp_z3 = []
    
        while not done:
            with utils.eval_mode(agent):
                action, z = agent.select_action(obs, eval=True)
                z1mQ_list.append(z['z1m_Q'])
                z2mQ_list.append(z['z2m_Q'])
                z3mQ_list.append(z['z3m_Q'])
                action_list.append(action)
                    
            if testpsnr:
                # compute PSNR
                obs_from_z1 = agent.decoder.get_obs_from_z1(z['z1m_Q']/qz[0])
                obs_from_z2 = agent.decoder.get_obs_from_z2(z['z2m_Q']/qz[1])
                obs_from_z3 = agent.decoder.get_obs_from_z3(z['z3m_Q']/qz[2])

                psnr_z1 = psnr(obs, (utils.depreprocess_obs(obs_from_z1.squeeze(0).cpu())).byte().numpy())
                psnr_z2 = psnr(obs, (utils.depreprocess_obs(obs_from_z2.squeeze(0).cpu())).byte().numpy())
                psnr_z3 = psnr(obs, (utils.depreprocess_obs(obs_from_z3.squeeze(0).cpu())).byte().numpy())
                episode_psnr_z1.append(psnr_z1)
                episode_psnr_z2.append(psnr_z2)
                episode_psnr_z3.append(psnr_z3)
                if q:
                    # bpp
                    episode_bpp_z1.append(z['z1_bpp'].cpu())
                    episode_bpp_z2.append(z['z2_bpp'].cpu())
                    episode_bpp_z3.append(z['z3_bpp'].cpu())

            obs, reward, done, _ = env.step(action)
            nextobs_list.append(obs)
            video.record(env)
            episode_reward += reward

        video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
        if testpsnr:
            L.log('eval/episode_psnr_z1', np.mean(episode_psnr_z1), step)
            L.log('eval/episode_psnr_z2', np.mean(episode_psnr_z2), step)
            L.log('eval/episode_psnr_z3', np.mean(episode_psnr_z3), step)
            if q:
                L.log('eval/episode_bpp_z1', np.mean(episode_bpp_z1), step)
                L.log('eval/episode_bpp_z2', np.mean(episode_bpp_z2), step)
                L.log('eval/episode_bpp_z3', np.mean(episode_bpp_z3), step)

    L = pred_act(z1mQ_list, z2mQ_list, z3mQ_list, action_list, nextobs_list, L, step, agent)

    L.dump(step)


def pred_act(z1mQ_list, z2mQ_list, z3mQ_list, action_list, nextobs_list, L, step, agent):
    ssa2 = pa_z(z2mQ_list, action_list, nextobs_list, agent, 1)
    ssa3 = pa_z(z3mQ_list, action_list, nextobs_list, agent, 2)
    L.log('eval/ssa2', ssa2, step)
    L.log('eval/ssa3', ssa3, step)
    return L


def pa_z(z, act, next_obs, agent, l):
    next_obs = torch.FloatTensor(numpy.array(next_obs)).to(agent.device)
    ssa_total = 0
    train_dataset = predDataset(z, act, next_obs, transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    with utils.eval_mode(agent):
        for i, (zmQ, actions, nob) in enumerate(train_loader):
            with torch.no_grad():
                zmQ = zmQ.float().cuda()
                actions = actions.float().cuda()
                nob = nob.float().cuda()
                states = zmQ
                feat = F.normalize(agent.critic.heads[l](states), dim=1)
                with torch.no_grad():
                    _, policy_action, log_pi, _ = agent.actor(nob)
                    target_Q1, target_Q2, _ = agent.critic_target(nob, policy_action)
                    target_V = torch.min(target_Q1,
                                        target_Q2) - agent.alpha.detach() * log_pi
                ssa = agent.constras_criterion(feat, actions, target_V.detach(), agent.device)
                ssa_total += ssa.detach().cpu().numpy()
    ssa_total /= len(train_loader)
    return ssa_total


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'sac_ae':
        return SacAeAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_latent_lambda=args.decoder_latent_lambda,
            decoder_weight_lambda=args.decoder_weight_lambda,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            qz=qz
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main():
    args = parse_args()

    # args.seed = np.random.randint(0, 1000)
    # print(args.seed)

    utils.set_seed_everywhere(args.seed)

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.image_size,
        width=args.image_size,
        frame_skip=args.action_repeat
    )
    env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )
    
    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_train_steps):
        if done:
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            if step % args.eval_freq == 0:
                L.log('eval/episode', episode, step)
                evaluate(env, agent, video, args.num_eval_episodes, L, step, testpsnr=args.testpsnr, q=args.q)
                if args.save_model:
                    agent.save(model_dir, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    main()
