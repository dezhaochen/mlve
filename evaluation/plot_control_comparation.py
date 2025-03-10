import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns;sns.set()
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

"""
comparation study
"""


def read_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def load_single_data(file_name):
    file = read_file(file_name)
    data = []
    for i in range(len(file)):
        if file[i].find('ER: ') != -1:
            d = file[i].split('ER: ')[1]
            idx = d.index('.')
            num = float(d[0:idx+5])
            data.append(num)
    # return data[0:40]
    return moving_average(data[0:40], 3)


def get_one_data(method, quality, data_seed):
    rewards = np.array([])
    for seed in data_seed:
        data = load_single_data('data/trainlog/discrete/%s/%s/%s.log' % (method, quality, seed))
        if rewards.size == 0:
            rewards = np.array(data)
        else:
            rewards = np.vstack((rewards, data))
    return rewards


def get_data():
    jpeg_1 = get_one_data('jpeg', '1', ['quality_1_8', 'quality_1_46_z', 'quality_1_129', 'quality_1_180_h',
                                        'quality_1_425_z', 'quality_1_427_z', 'quality_1_479_z', 'quality_1_550_h',
                                        'quality_1_579_z', 'quality_1_610'])
    jpeg_3 = get_one_data('jpeg', '3', ['quality_3_33', 'quality_3_76_h', 'quality_3_101_z', 'quality_3_220',
                                        'quality_3_306_z', 'quality_3_332_z', 'quality_3_367_h', 'quality_3_404_h',
                                        'quality_3_418_z', 'quality_3_449_z'])
    jpeg_5 = get_one_data('jpeg', '5', ['quality_5_132_z', 'quality_5_137_h', 'quality_5_139_z', 'quality_5_174_h',
                                        'quality_5_247_h', 'quality_5_398', 'quality_5_428_h', 'quality_5_524_z',
                                        'quality_5_557', 'quality_5_605_z'])
    jpeg_10 = get_one_data('jpeg', '10', ['quality_10_10_z', 'quality_10_49', 'quality_10_71', 'quality_10_96_z',
                                          'quality_10_203', 'quality_10_229_z', 'quality_10_306', 'quality_10_352_h',
                                          'quality_10_531_z', 'quality_10_566_z'])
    jpeg_20 = get_one_data('jpeg', '20', ['quality_20_52_z', 'quality_20_180', 'quality_20_239', 'quality_20_250_h',
                                          'quality_20_325', 'quality_20_419_z', 'quality_20_589_z', 'quality_20_679',
                                          'quality_20_745_z', 'quality_20_746'])
    jpeg_30 = get_one_data('jpeg', '30', ['quality_30_13_z', 'quality_30_97_z', 'quality_30_188_h', 'quality_30_255_h',
                                          'quality_30_285_h', 'quality_30_326_z', 'quality_30_356_z', 'quality_30_371',
                                          'quality_30_430', 'quality_30_445_h'])
    jpeg_40 = get_one_data('jpeg', '40', ['quality_40_30', 'quality_40_80', 'quality_40_163', 'quality_40_165_h',
                                          'quality_40_180_z', 'quality_40_181_z', 'quality_40_246', 'quality_40_449_z',
                                          'quality_40_549_z', 'quality_40_595_h'])
    jpeg_50 = get_one_data('jpeg', '50', ['quality_50_14', 'quality_50_248_z', 'quality_50_279_z', 'quality_50_285_h',
                                          'quality_50_314', 'quality_50_419', 'quality_50_425_h', 'quality_50_527_h',
                                          'quality_50_527_z', 'quality_50_606_h'])
    compressai_pre_1 = get_one_data('CompressAI-pretrained', '1', ['quality_1_52', 'quality_1_84_z', 'quality_1_234_z',
                                                                   'quality_1_269', 'quality_1_285', 'quality_1_300_z',
                                                                   'quality_1_411_h', 'quality_1_569_h',
                                                                   'quality_1_812_h', 'quality_1_828_h'])
    compressai_pre_2 = get_one_data('CompressAI-pretrained', '2', ['quality_2_24_z', 'quality_2_128', 'quality_2_197',
                                                                   'quality_2_204_z', 'quality_2_223_z',
                                                                   'quality_2_295_h', 'quality_2_418_h',
                                                                   'quality_2_532_h', 'quality_2_589_h',
                                                                   'quality_2_621'])
    compressai_pre_3 = get_one_data('CompressAI-pretrained', '3', ['quality_3_30_z', 'quality_3_72_h',
                                                                   'quality_3_256_z', 'quality_3_314',
                                                                   'quality_3_373_h', 'quality_3_406_h',
                                                                   'quality_3_530_z', 'quality_3_592',
                                                                   'quality_3_909_h', 'quality_3_929'])
    compressai_pre_4 = get_one_data('CompressAI-pretrained', '4', ['quality_4_383', 'quality_4_468_h',
                                                                   'quality_4_469_z', 'quality_4_545_z',
                                                                   'quality_4_562_h', 'quality_4_593',
                                                                   'quality_4_646', 'quality_4_858_h',
                                                                   'quality_4_27_h', 'quality_4_325_z'])
    compressai_pre_5 = get_one_data('CompressAI-pretrained', '5', ["quality_5_254_h", "quality_5_310_z",
                                                                   "quality_5_438_z", "quality_5_546",
                                                                   "quality_5_563_h", "quality_5_653",
                                                                   "quality_5_671_h", "quality_5_9", "quality_5_10",
                                                                   "quality_5_13_h"])
    compressai_pre_6 = get_one_data('CompressAI-pretrained', '6', ["quality_6_449", "quality_6_521_z",
                                                                   "quality_6_651_h", "quality_6_715",
                                                                   "quality_6_757_z", "quality_6_66_h", "quality_6_82",
                                                                   "quality_6_299_h", "quality_6_302_z",
                                                                   "quality_6_406_z"])
    ours_100 = get_one_data('ours', '100', ['bpp-100-5-task-6-4-KL-22-2_32_z', 'bpp-100-5-task-6-4-KL-22-2_71_z',
                                            'bpp-100-5-task-6-4-KL-22-2_179', 'bpp-100-5-task-6-4-KL-22-2_194_h',
                                            'bpp-100-5-task-6-4-KL-22-2_267', 'bpp-100-5-task-6-4-KL-22-2_515',
                                            'bpp-100-5-task-6-4-KL-22-2_677_h', 'bpp-100-5-task-6-4-KL-22-2_890',
                                            'bpp-100-5-task-6-4-KL-22-2_910', 'bpp-100-5-task-6-4-KL-22-2_993_z'])
    ours_1000 = get_one_data('ours', '1000', ['bpp-1000-10-task-10-4-KL-2-22_30_z',
                                             'bpp-1000-10-task-10-4-KL-2-22_86', 'bpp-1000-10-task-10-4-KL-2-22_96_h',
                                             'bpp-1000-10-task-10-4-KL-2-22_266_h', 'bpp-1000-10-task-10-4-KL-2-22_276',
                                             'bpp-1000-10-task-10-4-KL-2-22_288_h', 'bpp-1000-10-task-10-4-KL-2-22_374',
                                              'bpp-1000-10-task-10-4-KL-2-22_458',
                                              'bpp-1000-10-task-10-4-KL-2-22_489_h',
                                              'bpp-1000-10-task-10-4-KL-2-22_592_h'])
    MLVE = get_one_data('MLVE', 'MLVE', ['bpp-400-8-task-8-4-KL-6-18_47', 'bpp-500-8-task-9-4-KL-6-18_h',
                                      'bpp-800-9-task-10-4-KL-2-22_h', 'bpp-400-7-task-8-4-KL-10-14_z',
                                      'bpp-200-6-task-7div5-4-KL-14-10_z', 'bpp-200-6-task-8-4_23_h',
                                      'bpp-200-6-task-8-4_1_z', 'bpp-800-9-task-10-4-KL-2-22_48',
                                      'bpp-200-6-task-8-4_95_h', 'bpp-200-6-task-8-4_120_z'])
    # return [ours_100, jpeg_5, jpeg_10, jpeg_20, compressai_pre_1, compressai_pre_3, compressai_pre_5]
    return [MLVE, ours_1000, jpeg_5, jpeg_20, compressai_pre_1, compressai_pre_5]


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def main():
    plt.figure(figsize=(5, 5))
    data = get_data()[0:1]
    label = ['Ours_1']
    df1 = []
    for i in range(len(data)):
        df1.append(pd.DataFrame(data[i]).melt(var_name='Steps', value_name='Reward'))
        df1[i]['algo'] = label[i]
    df1 = pd.concat(df1)
    df1.index = range(len(df1))
    df1['Steps'] = df1['Steps'] * 10000
    palette1 = [sns.color_palette("husl", 6)[4]]

    sns.set_style("white")
    sns.plotting_context("paper")
    ax = sns.lineplot(x="Steps", y="Reward", hue="algo", style="algo", data=df1, palette=palette1, dashes=False)

    data = get_data()[2:4]
    label = ['JPEG_1', 'JPEG_10', 'JPEG_30']
    df1 = []
    for i in range(len(data)):
        df1.append(pd.DataFrame(data[i]).melt(var_name='Steps', value_name='Reward'))
        df1[i]['algo'] = label[i]
    df1 = pd.concat(df1)
    df1.index = range(len(df1))
    df1['Steps'] = df1['Steps'] * 10000
    palette1 = sns.color_palette("husl", 6)[2:4]

    sns.set_style("white")
    sns.plotting_context("paper")
    ax = sns.lineplot(x="Steps", y="Reward", hue="algo", style="algo", data=df1, palette=palette1, dashes=False,
                      linestyle='dotted')

    data = get_data()[4:6]
    label = ['CompressAI_pretrained_1', 'CompressAI_pretrained_3', 'CompressAI_pretrained_5']
    df1 = []
    for i in range(len(data)):
        df1.append(pd.DataFrame(data[i]).melt(var_name='Steps', value_name='Reward'))
        df1[i]['algo'] = label[i]
    df1 = pd.concat(df1)
    df1.index = range(len(df1))
    df1['Steps'] = df1['Steps'] * 10000
    palette1 = [sns.color_palette("husl", 6)[0], sns.color_palette("husl", 6)[5]]

    sns.set_style("white")
    sns.plotting_context("paper")
    ax = sns.lineplot(x="Steps", y="Reward", hue="algo", style="algo", data=df1, palette=palette1, dashes=False, linestyle='--')

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))

    palette1 = sns.color_palette("husl", 6)
    legend_elements = [
        Line2D([0], [0], color=palette1[4], label='MLVE_0.006'),
        Line2D([0], [0], color=palette1[1], label='JPEG_0.92', linestyle='dotted'),
        Line2D([0], [0], color=palette1[2], label='JPEG_1.13', linestyle='dotted'),
        Line2D([0], [0], color=palette1[0], label='CompressAI_0.17', linestyle='--'),
        Line2D([0], [0], color=palette1[5], label='CompressAI_0.47', linestyle='--'),
    ]
    plt.legend(handles=legend_elements, prop={'size': 14})

    handles, labels = ax.get_legend_handles_labels()

    # plt.title("Cartpole, swingup")
    ax.grid(True, linestyle='-', linewidth=0.5, color='#cccccc')
    plt.xlabel("Steps", fontsize=13)
    plt.ylabel("Reward", fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('output.png', dpi=300)
    plt.show()

    # ax.legend_.remove()


if __name__ == '__main__':
    main()