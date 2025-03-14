import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder_bilayer_Q import make_encoder
from decoder_bilayer import make_decoder

from loss import ContrasLoss


LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters, qt
    ):
        super().__init__()

        self.qt = qt

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False, eval=False
    ):
        N, _, H, W = obs.size()
        num_pixels = N * H * W * 3

        z1_mean, _, z2_mean, _, z3_mean, _ = self.encoder(obs, detach=detach_encoder)
        
        z3m_Q, z3m_likelihoods = self.encoder.entropy_bottleneck[2](z3_mean * self.qt[2])
        z3m_Q = z3m_Q / self.qt[2]

        if detach_encoder:
            z3m_Q = z3m_Q.detach()
            z3m_likelihoods = z3m_likelihoods.detach()

        mu, log_std = self.trunk(z3m_Q).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        if eval:
            with torch.no_grad():
                z, bpp = {}, {}
                for i in range(3):
                    z_key = f"z{i+1}m_Q"
                    bpp_key = f"z{i+1}_bpp"
                    z[z_key], likelihoods = self.encoder.entropy_bottleneck[i](locals()[f"z{i+1}_mean"] * self.qt[i])
                    bpp[bpp_key] = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            z.update({k: v.detach() for k, v in bpp.items()})
            return mu, pi, log_pi, log_std, z

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters, qt
    ):
        super().__init__()

        self.qt = qt

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(encoder_feature_dim, encoder_feature_dim),
                nn.PReLU(),
                nn.Linear(encoder_feature_dim, encoder_feature_dim)
            )
            for _ in range(3)
        ).cuda()

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        _, _, _, _, z3_mean, _ = self.encoder(obs, detach=detach_encoder)
        z3m_Q, z3m_likelihoods =  self.encoder.entropy_bottleneck[2](z3_mean * self.qt[2])
        z3m_Q = z3m_Q / self.qt[2]

        q1 = self.Q1(z3m_Q, action)
        q2 = self.Q2(z3m_Q, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2, z3m_likelihoods


class SacAeAgent(object):
    """SAC+AE algorithm."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        lambdaR = [1e-10, 1e-6, 1e-4],
        lambdaD=1e-6,
        lambdaE = [1e-8, 1e-4],
        qt=None,
        KLl=[22, 2],
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.lambdaR = lambdaR
        self.lambdaD = lambdaD
        self.lambdaE = lambdaE
        self.qt = qt
        self.KLl = KLl

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, qt
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, qt
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, qt
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.decoder = None
        if decoder_type != 'identity':
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(weight_init)

            net = self.critic.encoder
            parameters = {
                "net": {
                    name
                    for name, param in net.named_parameters()
                    if param.requires_grad and not name.endswith(".quantiles")
                },
                "aux": {
                    name
                    for name, param in net.named_parameters()
                    if param.requires_grad and name.endswith(".quantiles")
                },
            }

            # Make sure we don't have an intersection of parameters
            params_dict = dict(net.named_parameters())
            inter_params = parameters["net"] & parameters["aux"]
            union_params = parameters["net"] | parameters["aux"]
            assert len(inter_params) == 0
            assert len(union_params) - len(params_dict.keys()) == 0

            params_net = (params_dict[name] for name in sorted(parameters["net"]))
            params_aux = (params_dict[name] for name in sorted(parameters["aux"]))

            self.encoder_optimizer = torch.optim.Adam(
                params_net, lr=encoder_lr # parameters to update
            )
            self.aux_optimizer = torch.optim.Adam(
                params_aux, lr=1e-3 # parameters to exclude
            )

            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        cnet = self.critic
        cparameters = {
            "net": {
                name
                for name, param in cnet.named_parameters()
                if param.requires_grad and not name.endswith(".quantiles")
            },
            "aux": {
                name
                for name, param in cnet.named_parameters()
                if param.requires_grad and name.endswith(".quantiles")
            },
        }

        # Make sure we don't have an intersection of parameters
        cparams_dict = dict(cnet.named_parameters())
        inter_params = cparameters["net"] & cparameters["aux"]
        union_params = cparameters["net"] | cparameters["aux"]
        assert len(inter_params) == 0
        assert len(union_params) - len(cparams_dict.keys()) == 0

        cparams_net = (cparams_dict[name] for name in sorted(cparameters["net"]))
        cparams_aux = (cparams_dict[name] for name in sorted(cparameters["aux"]))

        self.critic_optimizer = torch.optim.Adam(
            cparams_net, lr=critic_lr, betas=(critic_beta, 0.999) # parameters to update
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

        self.constras_criterion = ContrasLoss()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, eval=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _, z = self.actor(
                obs, compute_pi=False, compute_log_pi=False, eval=eval
            )
            return mu.cpu().data.numpy().flatten(), z

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2, _ = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2, _ = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2, _ = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_decoder(self, obs, target_obs, L, step, next_obs):
        z1_mean, z1_sigma, z2_mean, z2_sigma, z3_mean, z3_sigma = self.critic.encoder(obs)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = utils.preprocess_obs(target_obs)

        # quantization encoding and decoding
        z1eps = torch.randn_like(z1_mean)
        z1m_Q, z1m_likelihoods =  self.critic.encoder.entropy_bottleneck[0](z1_mean*self.qt[0])
        z1 = z1m_Q / self.qt[0] + z1eps * z1_sigma
        z2eps = torch.randn_like(z2_mean)
        z2m_Q, z2m_likelihoods =  self.critic.encoder.entropy_bottleneck[1](z2_mean*self.qt[1])
        z2 = z2m_Q / self.qt[1] + z2eps * z2_sigma
        z3eps = torch.randn_like(z3_mean)
        z3m_Q, z3m_likelihoods =  self.critic.encoder.entropy_bottleneck[2](z3_mean*self.qt[2])
        z3 = z3m_Q / self.qt[2]

        # hvae loss
        rec_obs, pz1_mean, pz2_mean = self.decoder(z1, z2, z3)
        rec_loss = F.mse_loss(target_obs, rec_obs)
        KL_z1 = torch.mean(0.5 * torch.sum((z1_mean - pz1_mean) ** 2, dim=1), dim=0)
        KL_z2 = torch.mean(0.5 * torch.sum((z2_mean - pz2_mean) ** 2, dim=1), dim=0)
        hvae_loss = rec_loss + self.lambdaD * ((KL_z1 - self.KLl[0])**2 + (KL_z2 - self.KLl[1])**2)

        # bpp loss
        N, _, H, W = obs.size()
        num_pixels = N * H * W * 3
        z1_bpp_loss = (torch.log(z1m_likelihoods).sum() / (-math.log(2) * num_pixels))
        z2_bpp_loss = (torch.log(z2m_likelihoods).sum() / (-math.log(2) * num_pixels))
        z3_bpp_loss = (torch.log(z3m_likelihoods).sum() / (-math.log(2) * num_pixels))
        bpp_loss = self.lambdaR[0] * z1_bpp_loss + self.lambdaR[1] * z2_bpp_loss + self.lambdaR[2] * z3_bpp_loss

        # auxiliary loss
        aux_loss = self.critic.encoder.aux_loss()

        # ssa loss
        with torch.no_grad():
            _, approx_action, _, _ = self.actor(obs, compute_log_pi=False)
        state2 = z2m_Q
        state3 = z3m_Q
        feat2 = F.normalize(self.critic.heads[1](state2), dim=1)
        feat3 = F.normalize(self.critic.heads[2](state3), dim=1)
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2, _ = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
        ssa2 = self.constras_criterion(feat2, approx_action, target_V.detach(), self.device)
        ssa3 = self.constras_criterion(feat3, approx_action, target_V.detach(), self.device)
        ssa_loss = self.lambdaE[0] * ssa2 + self.lambdaE[1] * ssa3

        # total loss
        loss = hvae_loss + ssa_loss + bpp_loss
        
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.aux_optimizer.zero_grad()
        loss.backward()
        aux_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.aux_optimizer.step()
        L.log('train_ae/loss', loss, step)
        L.log('train_ae/recon', rec_loss, step)
        L.log('train_ae/kl2', KL_z2, step)
        L.log('train_ae/kl1', KL_z1, step)
        L.log('train_ae/bpp', bpp_loss, step)
        L.log('train_ae/aux', aux_loss, step)
        L.log('train_ae/z1_bpp', z1_bpp_loss, step)
        L.log('train_ae/z2_bpp', z2_bpp_loss, step)
        L.log('train_ae/z3_bpp', z3_bpp_loss, step)
        L.log('train_ae/ssa2', ssa2, step)
        L.log('train_ae/ssa3', ssa3, step)
        L.log('train_ae/conloss', ssa_loss, step)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if self.decoder is not None and step % self.decoder_update_freq == 0:
            self.update_decoder(obs, obs, L, step, next_obs)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        if self.decoder is not None:
            torch.save(
                self.decoder.state_dict(),
                '%s/decoder_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, step))
            )
