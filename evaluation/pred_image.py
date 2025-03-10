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
from decoder_bilayer import make_decoder

qz = [1000., 600., 10.]


"""
predict the true state of the environment by the trained model
"""


OUT_DIM = {2: 39, 4: 35, 6: 31}
class PrePixelModel(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]
        self.feature_dim = feature_dim
        self.fc = nn.Linear(feature_dim, 48672)
        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, obs_shape[0], 3, stride=2, output_padding=1
            )
        )

    def forward(self, z):
        h = torch.relu(self.fc(z))
        deconv = h.view(-1, self.num_filters, 39, 39)
        for i in range(4, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
        obs = self.deconvs[-1](deconv)
        return obs
    
    
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


class predDataset(Dataset):
    def __init__(self, data, label, transform=None):
        super(predDataset, self).__init__()
        self.img = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]
        if self.transform is not None:
            img = self.transform(img)
        img = np.squeeze(img['z1m_Q'])/qz[0]
        label = torch.from_numpy(label).float()
        return img, label


def test(test_loader, model, device):
    model.eval()
    # criterion = nn.MSELoss()
    loss = 0
    with torch.no_grad():
        for i, (img, label) in enumerate(test_loader):
            img = img.to(device)
            label = label.to(device)
            if label.dim() == 4:
                label = utils.preprocess_obs(label)
            output = model(img)
            loss += F.mse_loss(output, label)
    loss /= len(test_loader)
    print('test loss: {}'.format(loss.item()))
    return loss.item()


def train(train_loader, test_loader, model, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-7)#1e-3
    # criterion = nn.MSELoss()
    min_test_loss = 100
    for epoch in range(500):#200
        if epoch % 10 == 0:
            testloss = test(test_loader, model, device)
            if testloss < min_test_loss:
                min_test_loss = testloss
        model.train()
        for i, (img, label) in enumerate(train_loader):
            img = img.float().to(device)
            label = label.float().to(device)
            if label.dim() == 4:
                label = utils.preprocess_obs(label)
            optimizer.zero_grad()
            output = model(img)
            loss = F.mse_loss(output, label)
            loss.backward()
            optimizer.step()
            # if i % 100 == 0:
            #     print('epoch: {}, iter: {}, loss: {}'.format(epoch, i, loss.item()))
    print('min test loss: {}'.format(min_test_loss))


def main():
    print("start")
    args = parse_args()
    set_seed_everywhere(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('./data/test/bpp-600-7-task-7div5-4-KL-14-10_920000_obs.pkl', 'rb') as f:
        obs = pkl.load(f)
    f.close()
    with open('./data/test/bpp-600-7-task-7div5-4-KL-14-10_920000_z.pkl', 'rb') as f:
        z = pkl.load(f)
    f.close()

    train_state = obs[:int(len(obs)*0.8)]
    train_z = z[:int(len(z)*0.8)]
    test_state = obs[int(len(obs)*0.8):]
    test_z = z[int(len(z)*0.8):]

    train_dataset = predDataset(train_z, train_state, transform=None)
    test_dataset = predDataset(test_z, test_state, transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    decoder = PrePixelModel((9, 84, 84), feature_dim=args.encoder_feature_dim, num_layers=args.num_layers).to(device)
    train(train_loader, test_loader, decoder, device)


if __name__ == '__main__':
    main()
