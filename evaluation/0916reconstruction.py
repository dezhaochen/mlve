import numpy as np
import torch
import argparse
import os
import gym
import dmc2gym
import copy
import torch.nn.functional as F

import utils
import pickle as pkl
from torch.utils.data import Dataset

import torch.nn as nn
import random
from sac_ae_bilayer_ssa_bpp_0916 import SacAeAgent
import math
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

qz = [1000., 600., 10.]

"""
用来单独跑出重建出比较糊图像的编解码器
"""

OUT_DIM = {2: 39, 4: 35, 6: 31}
def CenterCrop(x, size):
    assert x.ndim == 4, 'input must be a 4D tensor'
    if x.size(2) == size and x.size(3) == size:
        return x
    assert x.size(3) == 100, f'unexpected size: {x.size(3)}'
    if size == 84:
        p = 8
    return x[:, :, p:-p, p:-p]


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


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
    parser.add_argument('--batch_size', default=128, type=int)
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
    parser.add_argument('--encoder_feature_dim', default=100, type=int)
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

    args = parser.parse_args()
    return args


class CustomDataset(Dataset):
    def __init__(self, obs, state, action):
        self.obs = obs
        self.state = state
        self.action = action

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        sample = {
            'obs': self.obs[idx],
            'action': self.action[idx]
        }
        return sample


class predDataset(Dataset):
    def __init__(self, data, label, transform=None):
        super(predDataset, self).__init__()
        self.img = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        img = np.squeeze(self.img[item])
        label = self.label[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def pa_z(z, act, agent, l):
    ssa_total = 0
    train_dataset = predDataset(z, act, transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    with utils.eval_mode(agent):
        for i, (zmQ, actions) in enumerate(train_loader):
            with torch.no_grad():
                zmQ = zmQ.float().cuda()
                actions = actions.float().cuda()
                states = zmQ
                feat = F.normalize(agent.critic.heads[l](states), dim=1)
                ssa = agent.constras_criterion(feat, actions, agent.device)
                ssa_total += ssa.detach().cpu().numpy()

    ssa_total /= len(train_loader)
    return ssa_total


def test(test_loader, agent, device, epoch):
    episode_psnr_z1 = []
    episode_psnr_z2 = []
    episode_psnr_z3 = []
    episode_bpp_z1 = []
    episode_bpp_z2 = []
    episode_bpp_z3 = []
    episode_ssa2 = []
    episode_ssa3 = []
    with utils.eval_mode(agent):
        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                obs = sample['obs'].float().to(device)
                action = sample['action'].float().to(device)

                z1mQ_list = []
                z2mQ_list = []
                z3mQ_list = []
                action_list = []

                for j in range(obs.shape[0]):
                    _, z = agent.select_action(obs[i], test=True)
                    z1mQ_list.append(z['z1m_Q'])
                    z2mQ_list.append(z['z2m_Q'])
                    z3mQ_list.append(z['z3m_Q'])
                    action_list.append(action[i])
                    # PSNR
                    obs_from_z1 = agent.decoder.get_obs_from_z1(z['z1m_Q'] / qz[0])
                    obs_from_z2 = agent.decoder.get_obs_from_z2(z['z2m_Q'] / qz[1])
                    obs_from_z3 = agent.decoder.get_obs_from_a(z['z3m_Q'] / qz[2])
                    psnr_z1 = psnr(obs[i].cpu().numpy().astype(np.uint8), (utils.depreprocess_obs(obs_from_z1.squeeze(0).cpu())).byte().numpy())
                    psnr_z2 = psnr(obs[i].cpu().numpy().astype(np.uint8), (utils.depreprocess_obs(obs_from_z2.squeeze(0).cpu())).byte().numpy())
                    psnr_z3 = psnr(obs[i].cpu().numpy().astype(np.uint8), (utils.depreprocess_obs(obs_from_z3.squeeze(0).cpu())).byte().numpy())
                    episode_psnr_z1.append(psnr_z1)
                    episode_psnr_z2.append(psnr_z2)
                    episode_psnr_z3.append(psnr_z3)

                    # bpp
                    episode_bpp_z1.append(z['z1_bpp'].cpu())
                    episode_bpp_z2.append(z['z2_bpp'].cpu())
                    episode_bpp_z3.append(z['z3_bpp'].cpu())

                # ssa
                ssa2 = pa_z(z2mQ_list, action_list, agent, 1)
                ssa3 = pa_z(z3mQ_list, action_list, agent, 2)
                episode_ssa2.append(ssa2)
                episode_ssa3.append(ssa3)

    # save model
    torch.save(agent.critic.encoder.state_dict(), f'./data/endec_model/encoder{epoch}.pth')
    torch.save(agent.decoder.state_dict(), f'./data/endec_model/decoder{epoch}.pth')
    print(f'psnr_z1: {np.mean(episode_psnr_z1):.4f} dB, psnr_z2: {np.mean(episode_psnr_z2):.4f} dB, psnr_z3: {np.mean(episode_psnr_z3):.4f} dB '
          f'bpp_z1: {np.mean(episode_bpp_z1):.4f}, bpp_z2: {np.mean(episode_bpp_z2):.4f}, bpp_z3: {np.mean(episode_bpp_z3):.4f} '
            f'ssa2: {np.mean(episode_ssa2):.4f}, ssa3: {np.mean(episode_ssa3):.4f}')


def train(train_loader, test_loader, agent, device):
    agent.train()
    for epoch in range(500):  # 200
        if epoch % 20 == 0 and epoch>0:
            testloss = test(test_loader, agent, device, epoch)

        agent.train()
        rec_list = []
        KL1_list = []
        KL2_list = []
        bpp1_list = []
        bpp2_list = []
        bpp3_list = []
        ssa2_list = []
        ssa3_list = []
        for i, sample in enumerate(train_loader):
            obs = sample['obs'].float().to(device)
            target_obs = obs.clone().detach()
            action = sample['action'].float().to(device)

            z1_mean, z1_sigma, z2_mean, z2_sigma, z3_mean, z3_sigma = agent.critic.encoder(obs)

            if target_obs.dim() == 4:
                # preprocess images to be in [-0.5, 0.5] range
                target_obs = utils.preprocess_obs(target_obs)

            # quantization encoding and decoding
            z1eps = torch.randn_like(z1_mean)
            z1m_Q, z1m_likelihoods = agent.critic.encoder.entropy_bottleneck1(z1_mean * agent.qz[0])
            z1 = z1m_Q / agent.qz[0] + z1eps * z1_sigma
            z2eps = torch.randn_like(z2_mean)
            z2m_Q, z2m_likelihoods = agent.critic.encoder.entropy_bottleneck2(z2_mean * agent.qz[1])
            z2 = z2m_Q / agent.qz[1] + z2eps * z2_sigma
            z3eps = torch.randn_like(z3_mean)
            z3m_Q, z3m_likelihoods = agent.critic.encoder.entropy_bottleneck3(z3_mean * agent.qz[2])
            a = z3m_Q / agent.qz[2] + z3eps * z3_sigma

            # hvae loss
            rec_obs, pz1_mean, pz2_mean = agent.decoder(z1, z2, a)
            rec_loss = F.mse_loss(target_obs, rec_obs)

            KL_z1 = torch.mean(0.5 * torch.sum((z1_mean - pz1_mean) ** 2, dim=1), dim=0)
            KL_z2 = torch.mean(0.5 * torch.sum((z2_mean - pz2_mean) ** 2, dim=1), dim=0)

            agent.decoder_latent_lambda = 1e-6
            hvae_loss = 1e3 * rec_loss + agent.decoder_latent_lambda * ((KL_z1 - 24) ** 2 + (KL_z2 - 6) ** 2)

            # bpp loss
            N, _, H, W = obs.size()
            num_pixels = N * H * W * 3
            z1_bpp_loss = (torch.log(z1m_likelihoods).sum() / (-math.log(2) * num_pixels))
            z2_bpp_loss = (torch.log(z2m_likelihoods).sum() / (-math.log(2) * num_pixels))
            z3_bpp_loss = (torch.log(z3m_likelihoods).sum() / (-math.log(2) * num_pixels))
            rlambda = [1e-10, 1e-4, 1e-4]
            bpp_loss = rlambda[0] * z1_bpp_loss + rlambda[1] * z2_bpp_loss + rlambda[2] * z3_bpp_loss

            # auxiliary loss
            aux_loss = agent.critic.encoder.aux_loss()

            # ssa loss
            states2 = z2m_Q
            states3 = z3m_Q
            feat2 = F.normalize(agent.critic.heads[1](states2), dim=1)
            feat3 = F.normalize(agent.critic.heads[2](states3), dim=1)
            ssa2 = agent.constras_criterion(feat2, action, agent.device)
            ssa3 = agent.constras_criterion(feat3, action, agent.device)
            ssa_loss = 1e-8 * ssa2 + 1e-4 * ssa3

            # total loss
            loss = hvae_loss + ssa_loss + bpp_loss

            rec_list.append(rec_loss.item())
            KL1_list.append(KL_z1.item())
            KL2_list.append(KL_z2.item())
            bpp1_list.append(z1_bpp_loss.item())
            bpp2_list.append(z2_bpp_loss.item())
            bpp3_list.append(z3_bpp_loss.item())
            ssa2_list.append(ssa2.item())
            ssa3_list.append(ssa3.item())

            agent.encoder_optimizer.zero_grad()
            agent.decoder_optimizer.zero_grad()
            agent.aux_optimizer.zero_grad()
            loss.backward()
            aux_loss.backward()
            agent.encoder_optimizer.step()
            agent.decoder_optimizer.step()
            agent.aux_optimizer.step()

        print(f'epoch: {epoch}, rec_loss: {np.mean(rec_list):.4f}, KL_z1: {np.mean(KL1_list):.4f}, KL_z2: {np.mean(KL2_list):.4f}, '
                f'bpp_z1: {np.mean(bpp1_list):.4f}, bpp_z2: {np.mean(bpp2_list):.4f}, bpp_z3: {np.mean(bpp3_list):.4f}, '
                f'ssa2: {np.mean(ssa2_list):.4f}, ssa3: {np.mean(ssa3_list):.4f}')


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
    args = parse_args()
    set_seed_everywhere(1)

    with open('./data/collected_data/bpp-100-5-task-6-4-KL-22-2_48_880000_obs.pkl', 'rb') as f:
        obs = pkl.load(f)
    f.close()
    with open('./data/collected_data/bpp-100-5-task-6-4-KL-22-2_48_880000_state.pkl', 'rb') as f:
        state = pkl.load(f)
    f.close()
    with open('./data/collected_data/bpp-100-5-task-6-4-KL-22-2_48_880000_act.pkl', 'rb') as f:
        action = pkl.load(f)
    f.close()

    # image = obs[10]
    # # show images by cv2
    # image = np.transpose(image[0:3, :, :], (1, 2, 0))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    train_dataset = CustomDataset(obs[:int(len(obs) * 0.8)], state[:int(len(obs) * 0.8)], action[:int(len(obs) * 0.8)])
    test_dataset = CustomDataset(obs[int(len(obs) * 0.8):], state[int(len(obs) * 0.8):], action[int(len(obs) * 0.8):])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )

    train(train_loader, test_loader, agent, device)


if __name__ == '__main__':
    main()
