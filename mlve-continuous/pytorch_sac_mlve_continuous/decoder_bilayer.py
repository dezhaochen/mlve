import torch
import torch.nn as nn

# from encoder import OUT_DIM
OUT_DIM = {2: 39, 4: 35, 6: 31}

"""
    reconstruct image
"""

class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]
        self.feature_dim = feature_dim

        self.fc1 = nn.Linear(feature_dim, 48672)
        self.fc2 = nn.Linear(feature_dim, 39200)
        self.fa = nn.Linear(feature_dim, num_filters * self.out_dim * self.out_dim)

        self.z1_mean = nn.Sequential(nn.Linear(48672, self.feature_dim),
                                    nn.LayerNorm(self.feature_dim, elementwise_affine=False))
        self.z2_mean = nn.Sequential(nn.Linear(39200, self.feature_dim),
                                    nn.LayerNorm(self.feature_dim, elementwise_affine=False))

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

        self.outputs = dict()

    def get_obs_from_z1(self, z1):
        h1 = torch.relu(self.fc1(z1))
        deconv = h1.view(-1, self.num_filters, 39, 39)
        for i in range(4, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
        obs = self.deconvs[-1](deconv)
        return obs
    
    def get_obs_from_z2(self, z2):
        h2 = self.fc2(z2)
        deconv = h2.view(-1, self.num_filters, 35, 35)
        for i in range(2, 4):
            deconv = torch.relu(self.deconvs[i](deconv))
        deconv = deconv.view(deconv.size(0), -1)
        z1_mean = self.z1_mean(deconv)
        obs = self.get_obs_from_z1(z1_mean)
        return obs
    
    def get_obs_from_a(self, a):
        h3 = self.fa(a)
        deconv = h3.view(-1, self.num_filters, self.out_dim, self.out_dim)
        for i in range(0, 2):
            deconv = torch.relu(self.deconvs[i](deconv))
        deconv = deconv.view(deconv.size(0), -1)
        z2_mean = self.z2_mean(deconv)
        obs = self.get_obs_from_z2(z2_mean)
        return obs

    def forward(self, z1, z2, a):
        # p(x|z1)
        h1 = torch.relu(self.fc1(z1))
        deconv = h1.view(-1, self.num_filters, 39, 39)
        for i in range(4, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
        obs = self.deconvs[-1](deconv)

        # p(z1|z2)
        h2 = self.fc2(z2)
        deconv = h2.view(-1, self.num_filters, 35, 35)
        for i in range(2, 4):
            deconv = torch.relu(self.deconvs[i](deconv))
        deconv = deconv.view(deconv.size(0), -1)
        z1_mean = self.z1_mean(deconv)

        # p(z2|a)
        h3 = self.fa(a)
        deconv = h3.view(-1, self.num_filters, self.out_dim, self.out_dim)
        for i in range(0, 2):
            deconv = torch.relu(self.deconvs[i](deconv))
        deconv = deconv.view(deconv.size(0), -1)
        z2_mean = self.z2_mean(deconv)

        return obs, z1_mean, z2_mean

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param(
                'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
            )
        L.log_param('train_decoder/fc', self.fc, step)


_AVAILABLE_DECODERS = {'pixel': PixelDecoder}


def make_decoder(
    decoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert decoder_type in _AVAILABLE_DECODERS
    return _AVAILABLE_DECODERS[decoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )
