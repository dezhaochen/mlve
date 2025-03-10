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

import utils
import pickle as pkl

from sac_ae_pixel import SacAeAgent
from collections import deque
from itertools import islice


W = 250 * 10**3
K = 52
L = K * 50
N = 4
kappa = 20
a = W / L - N * kappa


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cartpole')
    parser.add_argument('--task_name', default='swingup')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # train
    parser.add_argument('--agent', default='sac_ae', type=str)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
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

    parser.add_argument('--actor_path', default='./data/models/pixel_actor_420000.pt', type=str)
    parser.add_argument('--critic_path', default='./data/models/pixel_critic_420000.pt', type=str)

    args = parser.parse_args()
    return args


def transmission_exp(env, agent):
    step = 0
    obs = env.reset()
    done = False
    episode_reward = 0
    action_queue = deque(maxlen=200)  # 动作历史队列
    action_queue.append(np.zeros(env.action_space.shape))  # 初始化动作队列
    transmission_queue = deque()  # 模拟传输中的包队列，每个元素是 (动作, 传输完成时间)
    current_action = action_queue[-1]  # 当前执行的动作

    while not done:
        with utils.eval_mode(agent):  # 评估模式下选择动作
            action = agent.select_action(obs)

        # 当前时间步传输时延采样值
        sample = np.random.exponential(1 / a)

        # 如果采样值小于 0.01，则立即传输最新动作
        if sample < 0.01:
            transmission_queue.clear()  # 丢弃所有未到达的包
            transmission_queue.append((action, step))  # 立即传输，立即到达
        else:
            # 否则，将包加入传输队列，根据延迟时间计算预计接收时间
            delay_steps = int(sample * 100)  # 映射延迟时间
            received_time = step + delay_steps  # 预计接收时间

            # 丢弃所有到达时间晚于或等于新包的旧包
            transmission_queue = deque(
                [(act, time) for act, time in transmission_queue if time < received_time]
            )
            # 将新包添加在最后
            transmission_queue.append((action, received_time))

        # 检查是否有包到达当前时间步
        if transmission_queue and transmission_queue[0][1] == step:
            current_action = transmission_queue.popleft()[0]  # 取出最早到达的包

        # 使用当前有效的动作
        action_queue.append(current_action)

        # 执行动作并获取新状态和奖励
        obs, reward, done, _ = env.step(current_action)
        episode_reward += reward
        step += 1

    return episode_reward


def evaluate(env, agent, num_episodes):
    episode_reward_list = []
    with torch.no_grad():
        for i in range(num_episodes):
            episode_reward = transmission_exp(env, agent)
            episode_reward_list.append(episode_reward)
            # print(episode_reward)
            print("env reward: " + str(episode_reward))
    return episode_reward_list


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
            num_filters=args.num_filters
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main():
    print("start")
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

    # evaluate agent periodically
    episode_reward_list = evaluate(env, agent, args.num_eval_episodes)
    print('average envs reward:', sum(episode_reward_list) / len(episode_reward_list))


if __name__ == '__main__':
    main()
