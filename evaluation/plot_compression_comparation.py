import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns;sns.set()
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D


my_psnr = [31.9663, 32.6202, 33.6169, 34.3482, 34.7722, 35.0395]
my_bpp = [0.0212, 0.0357, 0.0431, 0.0585, 0.1181, 0.1271]
jpeg_psnr = [29.8133, 31.5524, 33.1535, 34.2998, 34.8503]
jpeg_bpp = [0.9249, 1.0255, 1.1276, 1.2451, 1.3355]
bpg_psnr = [31.0953, 32.5364, 32.9599, 33.8359, 34.6177, 34.9465]
bpg_bpp = [0.0778, 0.1295, 0.1448, 0.1775, 0.2224, 0.2483]
compressai_psnr = [29.3062, 30.9259, 32.2622, 34.0288, 34.8562]
compressai_bpp = [0.1659, 0.2043, 0.2870, 0.3545, 0.4700]

data = {'my_psnr': my_psnr, 'my_bpp': my_bpp, 'jpeg_psnr': jpeg_psnr, 'jpeg_bpp': jpeg_bpp,
        'compressai_psnr': compressai_psnr, 'compressai_bpp': compressai_bpp}

# df = pd.DataFrame(data)

sns.set_style("white")
sns.plotting_context("paper")
plt.plot(my_bpp, my_psnr, linestyle='-')
plt.scatter(my_bpp, my_psnr, color='w', marker='v', edgecolors='#1f77b4', s=80, linewidths=1.5)
plt.plot(compressai_bpp, compressai_psnr, linestyle='-')
plt.scatter(compressai_bpp, compressai_psnr, color='w', marker='o', edgecolors='#ff7f0e', s=80, linewidths=1.5)
plt.plot(bpg_bpp, bpg_psnr, linestyle='-', color='r')
plt.plot(jpeg_bpp, jpeg_psnr, linestyle='-')
plt.scatter(jpeg_bpp, jpeg_psnr, color='w', marker='s', edgecolors='#2ca02c', s=80, linewidths=1.5)
plt.scatter(bpg_bpp, bpg_psnr, color='w', marker='o', edgecolors='r', s=80, linewidths=1.5)
ax = plt.gca()
plt.xlabel("Rit Rate [bit/px]")
plt.ylabel("PSNR↑ [dB]")
ax.grid(True, linestyle='-', linewidth=0.5, color='#cccccc')
plt.show()
