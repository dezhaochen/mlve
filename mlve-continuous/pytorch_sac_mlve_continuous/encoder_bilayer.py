import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


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
        self.z2_mean = nn.Sequential(nn.Linear(39200, self.feature_dim),
                                    nn.LayerNorm(self.feature_dim, elementwise_affine=False))  # 均值
        self.a_mean = nn.Sequential(nn.Linear(num_filters * out_dim * out_dim, self.feature_dim),
                                    nn.LayerNorm(self.feature_dim, elementwise_affine=False))

        # self.sigmas = torch.tensor([0.1000, 0.3000, 1.0000]).cuda()
        self.sigmas = torch.tensor([0.1000, 0.5000, 1.0000]).cuda()

        self.outputs = dict()

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
        z1eps = torch.randn_like(z1_mean)
        z1 = z1_mean + z1eps*z1_sigma

        h2 = self.forward_conv2(h1)
        h2_ = h2.view(h2.size(0), -1)
        z2_mean = self.z2_mean(h2_)
        z2eps = torch.randn_like(z2_mean)
        z2 = z2_mean + z2eps*z2_sigma

        ha = self.forward_conv3(h2)
        a_mean = self.a_mean(ha)
        aeps = torch.randn_like(a_mean)
        a = a_mean + aeps*z3_sigma

        if detach:
            z1 = z1.detach()
            z1_mean = z1_mean.detach()
            z2 = z2.detach()
            z2_mean = z2_mean.detach()
            a = a.detach()
            a_mean = a_mean.detach()

        return z1, z1_mean, z1_sigma, z2, z2_mean, z2_sigma, a, a_mean, z3_sigma

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])
        tie_weights(src=source.z1_mean[0], trg=self.z1_mean[0])
        tie_weights(src=source.z2_mean[0], trg=self.z2_mean[0])
        tie_weights(src=source.a_mean[0], trg=self.a_mean[0])

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
