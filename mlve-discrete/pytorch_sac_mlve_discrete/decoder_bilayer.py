import torch
import torch.nn as nn

OUT_DIM = {2: 39, 4: 35, 6: 31}


class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = [OUT_DIM[2], OUT_DIM[4], OUT_DIM[6]]
        self.feature_dim = feature_dim

        self.fc = []
        self.z_mean = []
        for i in range(3):
            self.fc.append(nn.Linear(feature_dim, num_filters * self.out_dim[i] * self.out_dim[i]))
            if i !=2:
                # predict z1 and z2
                self.z_mean.append(nn.Sequential(nn.Linear(num_filters * self.out_dim[i] * self.out_dim[i], self.feature_dim),
                                    nn.LayerNorm(self.feature_dim, elementwise_affine=False)))

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
        h1 = torch.relu(self.fc[0](z1))
        deconv = h1.view(-1, self.num_filters, self.out_dim[0], self.out_dim[0])
        for i in range(4, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
        obs = self.deconvs[-1](deconv)
        return obs
    
    def get_obs_from_z2(self, z2, forward=False):
        h2 = self.fc[1](z2)
        deconv = h2.view(-1, self.num_filters, self.out_dim[1], self.out_dim[1])
        for i in range(2, 4):
            deconv = torch.relu(self.deconvs[i](deconv))
        deconv = deconv.view(deconv.size(0), -1)
        z1_mean = self.z_mean[0](deconv)
        if forward:
            return z1_mean
        obs = self.get_obs_from_z1(z1_mean)
        return obs
    
    def get_obs_from_z3(self, z3, forward=False):
        h3 = self.fc[2](z3)
        deconv = h3.view(-1, self.num_filters, self.out_dim[2], self.out_dim[2])
        for i in range(0, 2):
            deconv = torch.relu(self.deconvs[i](deconv))
        deconv = deconv.view(deconv.size(0), -1)
        z2_mean = self.z_mean[1](deconv)
        if forward:
            return z2_mean
        obs = self.get_obs_from_z2(z2_mean)
        return obs

    def forward(self, z1, z2, z3):
        # p(x|z1)
        obs = get_obs_from_z1(z1)
        # p(z1|z2)
        z1_mean = get_obs_from_z2(z2, forward=True)
        # p(z2|z3)
        z2_mean = get_obs_from_z3(z3, forward=True)
        return obs, z1_mean, z2_mean


_AVAILABLE_DECODERS = {'pixel': PixelDecoder}


def make_decoder(
    decoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert decoder_type in _AVAILABLE_DECODERS
    return _AVAILABLE_DECODERS[decoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )
