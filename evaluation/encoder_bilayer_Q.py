import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck
from typing import cast
from torch import Tensor


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def tie_entropy(src, trg):
    for name, _ in src.named_parameters():
        commmand = "trg." + name + " = src." + name
        exec(commmand)


OUT_DIM = {2: 39, 4: 35, 6: 31}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.z1_mean = nn.Sequential(nn.Linear(48672, self.feature_dim),
                                    nn.LayerNorm(self.feature_dim, elementwise_affine=False))  # 均值
        self.z1_logvar = nn.Sequential(nn.Linear(48672, self.feature_dim),
                                    nn.LayerNorm(self.feature_dim, elementwise_affine=False))  # 方差
        self.z2_mean = nn.Sequential(nn.Linear(39200, self.feature_dim),
                                    nn.LayerNorm(self.feature_dim, elementwise_affine=False))  # 均值
        self.z2_logvar = nn.Sequential(nn.Linear(39200, self.feature_dim),
                                    nn.LayerNorm(self.feature_dim, elementwise_affine=False))  # 方差
        self.a_mean = nn.Sequential(nn.Linear(num_filters * out_dim * out_dim, self.feature_dim),
                                    nn.LayerNorm(self.feature_dim, elementwise_affine=False))
        self.a_logvar = nn.Sequential(nn.Linear(num_filters * out_dim * out_dim, self.feature_dim),
                                    nn.LayerNorm(self.feature_dim, elementwise_affine=False))

        self.outputs = dict()

        self.entropy_bottleneck1 = EntropyBottleneck(feature_dim)
        self.entropy_bottleneck2 = EntropyBottleneck(feature_dim)
        self.entropy_bottleneck3 = EntropyBottleneck(feature_dim)
        self.sigmas = torch.tensor([1.0000, 1.0000, 1.0000]).cuda()

    def forward_conv1(self, obs):
        obs = obs / 255.
        conv = torch.relu(self.convs[0](obs))
        for i in range(1, 2):
            conv = torch.relu(self.convs[i](conv))
        return conv

    def forward_conv2(self, obs):
        conv = obs
        for i in range(2, 4):
            conv = torch.relu(self.convs[i](conv))
        return conv

    def forward_conv3(self, obs):
        conv = obs
        for i in range(4, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        z1_sigma = self.sigmas[2]
        z2_sigma = self.sigmas[1]
        z3_sigma = self.sigmas[0]

        h1 = self.forward_conv1(obs)
        h1_ = h1.view(h1.size(0), -1)
        z1_mean = self.z1_mean(h1_)

        h2 = self.forward_conv2(h1)
        h2_ = h2.view(h2.size(0), -1)
        z2_mean = self.z2_mean(h2_)

        ha = self.forward_conv3(h2)
        a_mean = self.a_mean(ha)

        if detach:
            z1_mean = z1_mean.detach()
            z2_mean = z2_mean.detach()
            a_mean = a_mean.detach()

        return z1_mean, z1_sigma, z2_mean, z2_sigma, a_mean, z3_sigma

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])
        tie_weights(src=source.z1_mean[0], trg=self.z1_mean[0])
        tie_weights(src=source.z1_logvar[0], trg=self.z1_logvar[0])
        tie_weights(src=source.z2_mean[0], trg=self.z2_mean[0])
        tie_weights(src=source.z2_logvar[0], trg=self.z2_logvar[0])
        tie_weights(src=source.a_mean[0], trg=self.a_mean[0])
        tie_weights(src=source.a_logvar[0], trg=self.a_logvar[0])

        tie_entropy(src=source.entropy_bottleneck1, trg=self.entropy_bottleneck1)
        tie_entropy(src=source.entropy_bottleneck2, trg=self.entropy_bottleneck2)
        tie_entropy(src=source.entropy_bottleneck3, trg=self.entropy_bottleneck3)

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)

    def aux_loss(self) -> Tensor:
        r"""Returns the total auxiliary loss over all ``EntropyBottleneck``\s.

        In contrast to the primary "net" loss used by the "net"
        optimizer, the "aux" loss is only used by the "aux" optimizer to
        update *only* the ``EntropyBottleneck.quantiles`` parameters. In
        fact, the "aux" loss does not depend on image data at all.

        The purpose of the "aux" loss is to determine the range within
        which most of the mass of a given distribution is contained, as
        well as its median (i.e. 50% probability). That is, for a given
        distribution, the "aux" loss converges towards satisfying the
        following conditions for some chosen ``tail_mass`` probability:

        * ``cdf(quantiles[0]) = tail_mass / 2``
        * ``cdf(quantiles[1]) = 0.5``
        * ``cdf(quantiles[2]) = 1 - tail_mass / 2``

        This ensures that the concrete ``_quantized_cdf``\s operate
        primarily within a finitely supported region. Any symbols
        outside this range must be coded using some alternative method
        that does *not* involve the ``_quantized_cdf``\s. Luckily, one
        may choose a ``tail_mass`` probability that is sufficiently
        small so that this rarely occurs. It is important that we work
        with ``_quantized_cdf``\s that have a small finite support;
        otherwise, entropy coding runtime performance would suffer.
        Thus, ``tail_mass`` should not be too small, either!
        """
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )
