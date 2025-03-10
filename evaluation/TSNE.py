import torch
import pickle as pkl

import numpy as np
import torch
import argparse
import os
# os.environ['MUJOCO_GL'] = 'egl'
import torch.nn.functional as F
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy

import utils
import pickle as pkl

from sac_ae_bilayer_ssa_bpp_test import SacAeAgent
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error

import cv2

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mlt
from matplotlib.colors import LinearSegmentedColormap


qz = [1000., 600., 10.]


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

    # parser.add_argument('--actor_path', default='./data/models/bpp-600-8-task-9-4-KL-6-18_actor_730000.pt', type=str)
    # parser.add_argument('--critic_path', default='./data/models/bpp-600-8-task-9-4-KL-6-18_critic_730000.pt', type=str)
    # parser.add_argument('--decoder_path', default='./data/models/bpp-600-8-task-9-4-KL-6-18_decoder_730000.pt', type=str)
    parser.add_argument('--actor_path', default='./data/models/bpp-100-5-task-6-4-KL-22-2_48_actor_880000.pt', type=str)
    parser.add_argument('--critic_path', default='./data/models/bpp-100-5-task-6-4-KL-22-2_48_critic_880000.pt', type=str)
    parser.add_argument('--decoder_path', default='./data/models/bpp-100-5-task-6-4-KL-22-2_48_decoder_880000.pt', type=str)

    args = parser.parse_args()
    return args


def reconstruction(agent, obs, state, z_, act):
    with torch.no_grad():
        z1_list = []
        z2_list = []
        z3_list = []
        act_list = []
        for id in range(len(obs)):
            o = obs[id]
            s = state[id]
            z = z_[id]

            z1_list.append(z['z1m_Q'] / qz[0])
            z2_list.append(z['z2m_Q'] / qz[1])
            z3_list.append(z['z3m_Q'] / qz[2])
            act_list.append(act[id])

        z1 = torch.stack(z1_list, 0)
        z1 = z1.view(z1.size(0), -1)
        z2 = torch.stack(z2_list, 0)
        z2 = z2.view(z2.size(0), -1)
        z2 = F.normalize(agent.critic.heads[1](z2), dim=1)
        z3 = torch.stack(z3_list, 0)
        z3 = z3.view(z3.size(0), -1)
        z3 = F.normalize(agent.critic.heads[2](z3), dim=1)
        act_list = np.array(act_list)
        act_list = act_list.squeeze()

        TSNE_plot(act_list[0:5000], z1[0:5000], p=30)
    return 0


def TSNE_plot(act, z, p):
    labels = act
    tsne = TSNE(n_components=3, init='pca', perplexity=p)
    result = tsne.fit_transform(z)
    x_min, x_max = np.min(result, 0), np.max(result, 0)
    data1 = (result - x_min) / (x_max - x_min)

    color_map = plt.cm.get_cmap('coolwarm')
    normalize = plt.Normalize(vmin=-1, vmax=1)
    fig = plt.figure()
    # ax = fig.add_subplot()
    # scatter = ax.scatter(data1[:, 0], data1[:, 1], c=labels, cmap=color_map, norm=normalize, s=2)
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c=labels, cmap=color_map, norm=normalize, s=2)
    bar = plt.colorbar(scatter, ax=ax, ticks=[-1.0, 0.0, 1.0])
    # bar.set_label('Z Value')

    plt.show()
    return 0


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
    with open('./data/collected_data/bpp-600-8-task-9-4-KL-6-18_47_730000_obs.pkl', 'rb') as f:
        obs = pkl.load(f)
    f.close()
    with open('./data/collected_data/bpp-600-8-task-9-4-KL-6-18_47_730000_state.pkl', 'rb') as f:
        state = pkl.load(f)
    f.close()
    with open('./data/collected_data/bpp-600-8-task-9-4-KL-6-18_47_730000_z.pkl', 'rb') as f:
        z = pkl.load(f)
    f.close()
    with open('./data/collected_data/bpp-600-8-task-9-4-KL-6-18_47_730000_act.pkl', 'rb') as f:
        act = pkl.load(f)
    f.close()

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

    # load trained weights
    agent.actor.load_state_dict(torch.load(args.actor_path, map_location=device))
    agent.critic.load_state_dict(torch.load(args.critic_path, map_location=device))
    agent.decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))

    reconstruction(agent, obs, state, z, act)


if __name__ == '__main__':
    main()

