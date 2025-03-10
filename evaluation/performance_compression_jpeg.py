from PIL import Image
import os
import pickle as pkl
import numpy as np
import math


def main():
    with open('./data/test/bpp-100-5-task-6-4-KL-22-2_880000_obs.pkl', 'rb') as f:
        image = pkl.load(f)
    f.close()

    quality = 40

    bpp_list = []
    psnr_list = []
    for i in range(len(image)):
        img = Image.fromarray(np.transpose(image[i][0:3, :, :],  (1, 2, 0)))
        width, height = img.size
        num_channels = len(img.getbands())
        img.save('n.jpg', 'JPEG', quality=quality)
        # 重新加载压缩后的图片以计算PSNR
        with Image.open('n.jpg') as compressed_img:
            # 计算文件大小和bpp
            file_size_bytes = os.path.getsize('n.jpg')
            total_pixels = width * height
            bpp = (file_size_bytes * 8) / total_pixels
            bpp_list.append(bpp)

            # 将图片转换为NumPy数组
            original = np.array(img)
            compressed = np.array(compressed_img)

            # 计算MSE
            mse = np.mean((original - compressed) ** 2)
            # 计算PSNR
            pixel_max = 255.0
            psnr = 20 * math.log10(pixel_max / math.sqrt(mse))
            psnr_list.append(psnr)
        compressed_img.close()
    print(f'Average PSNR: {np.mean(psnr_list):.4f}dB')
    print(f'Average Bit-rate: {np.mean(bpp_list):.4f} bpp')


if __name__ == '__main__':
    main()