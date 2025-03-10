import torch
import pickle as pkl

import numpy as np
import torch
import argparse
import os
# os.environ['MUJOCO_GL'] = 'egl'
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy
import cv2

import utils
import pickle as pkl

from sac_ae_bilayer_ssa_bpp_test import SacAeAgent
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error

import cv2

qz = [1000., 1000., 10.]

# 最后用于测试PSNR和跑重建图


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
    parser.add_argument('--init_steps', default=1000, type=int)  # 1000 0
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
    parser.add_argument('--encoder_feature_dim', default=100, type=int)  # 50
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

    parser.add_argument('--actor_path', default='./data/models/bpp-1000-10-task-10-4-KL-2-12_actor_550000.pt', type=str)
    parser.add_argument('--critic_path', default='./data/models/bpp-1000-10-task-10-4-KL-2-12_critic_550000.pt', type=str)
    parser.add_argument('--decoder_path', default='./data/models/bpp-1000-10-task-10-4-KL-2-12_decoder_550000.pt', type=str)

    args = parser.parse_args()
    return args


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
    print("start")
    with open('./data/test/bpp-100-5-task-6-4-KL-22-2_880000_obs.pkl', 'rb') as f:
        obs = pkl.load(f)
    f.close()
    # with open('./data/test/bpp-1000-10-task-10-4-KL-2-12_550000_state.pkl', 'rb') as f:
    #     state = pkl.load(f)
    # f.close()
    # with open('./data/test/bpp-1000-10-task-10-4-KL-2-12_550000_z.pkl', 'rb') as f:
    #     z = pkl.load(f)
    # f.close()

    args = parse_args()
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )

    psnr_list = []
    # 0917
    agent.critic.encoder.load_state_dict(torch.load('./data/0916/107z1encoder180.pth', map_location=device))
    agent.decoder.load_state_dict(torch.load('./data/0916/107z1decoder180.pth', map_location=device))
    for i in range(len(obs)):
        _, z = agent.select_action(obs[i], test=True)
        obs_from_z3 = agent.decoder.get_obs_from_a(z['z3m_Q'] / 10)
        aaa = (utils.depreprocess_obs(obs_from_z3.squeeze(0).cpu())).byte().numpy()
        image = np.transpose(aaa[0:3, :, :], (1, 2, 0))
        o = np.transpose(obs[i][0:3, :, :], (1, 2, 0))
        psnr = peak_signal_noise_ratio(o, image)
        print(psnr)
        psnr_list.append(psnr)

        # _, z = agent.select_action(obs[i], test=True)
        # obs_from_z1 = agent.decoder.get_obs_from_z1(z['z1m_Q'] / 1000)
        # aaa = (utils.depreprocess_obs(obs_from_z1.squeeze(0).cpu())).byte().numpy()
        # image = np.transpose(aaa[0:3, :, :], (1, 2, 0))
        # o = np.transpose(obs[i][0:3, :, :], (1, 2, 0))
        # peak_signal_noise_ratio(o, image)



if __name__ == '__main__':
    main()

