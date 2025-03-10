from PIL import Image
import os
import pickle as pkl
import numpy as np
import math
import subprocess


def main():
    with open('./data/test/bpp-100-5-task-6-4-KL-22-2_880000_obs.pkl', 'rb') as f:
        image = pkl.load(f)
    f.close()
    image = image[0:1000]

    quality = 43

    bpp_list = []
    psnr_list = []
    for i in range(len(image)):
        im = Image.fromarray(np.transpose(image[i][0:3, :, :],  (1, 2, 0)))
        # 保存img图片到本地
        im.save('in.png')

        # 使用bpgenc工具压缩图片
        subprocess.run(['/home/user/libbpg/bpgenc', '-o', 'out.bpg', '-q', str(quality), 'in.png'])

        # 使用bpgdec工具解码图片以计算PSNR
        decompressed_image_path = 'out.bpg'.replace('.bpg', '_decompressed.png')
        subprocess.run(['/home/user/libbpg/bpgdec', '-o', decompressed_image_path, 'out.bpg'])

        # 计算bpp
        file_size_bytes = os.path.getsize('out.bpg')
        with Image.open('in.png') as img:
            width, height = img.size
            total_pixels = width * height
            bpp = (file_size_bytes * 8) / total_pixels
        img.close()
        # 计算PSNR
        original = np.array(Image.open('in.png'))
        compressed = np.array(Image.open(decompressed_image_path))
        mse = np.mean((original - compressed) ** 2)
        pixel_max = 255.0
        psnr = 20 * math.log10(pixel_max / math.sqrt(mse))

        bpp_list.append(bpp)
        psnr_list.append(psnr)

        # 清理解码后的临时文件
        os.remove(decompressed_image_path)

    print(f'Average PSNR: {np.mean(psnr_list):.4f}dB')
    print(f'Average Bit-rate: {np.mean(bpp_list):.4f} bpp')


if __name__ == '__main__':
    main()