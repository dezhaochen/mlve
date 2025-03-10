import math
import io
import torch
from compressai.zoo import cheng2020_anchor
import pickle as pkl
from utils import pad
import numpy as np
from torch.utils.data import Dataset


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()


class myDataset(Dataset):
    def __init__(self, data, transform=None):
        super(myDataset, self).__init__()
        self.img = data
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        img = np.squeeze(self.img[item])
        if self.transform is not None:
            img = self.transform(img)
        return img


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = cheng2020_anchor(quality=1, pretrained=True).eval().to(device)
    with open('./data/test/bpp-100-5-task-6-4-KL-22-2_880000_obs.pkl', 'rb') as f:
        img = pkl.load(f)
    f.close()
    test_dataset = myDataset(img)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    PSNR_list = []
    BPP_list = []
    for i, img in enumerate(test_loader):
        img = img[:, 0:3, :, :].to(device)/255
        x = pad(img, p=2**6)
        with torch.no_grad():
            out_net = net.forward(x)
        out_net['x_hat'].clamp_(0, 1)
        out_net['x_hat'] = out_net['x_hat'][:, :, 22:106, 22:106]
        PSNR_list.append(compute_psnr(img, out_net['x_hat']))
        BPP_list.append(compute_bpp(out_net))

    print(f'Average PSNR: {np.mean(PSNR_list):.2f}dB')
    print(f'Average Bit-rate: {np.mean(BPP_list):.3f} bpp')


if __name__ == '__main__':
    main()