import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns;sns.set()
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

"""
ablation study
"""


def read_file(file_name):
    with open(file_name, 'r') as f:
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
    return moving_average(data[0:42], 3)


def get_one_data(method, data_seed):
    rewards = np.array([])
    for seed in data_seed:
        data = load_single_data('data/trainlog/continuous/%s/%s.log' % (method, seed))
        if rewards.size == 0:
            rewards = np.array(data)
        else:
            rewards = np.vstack((rewards, data))
    return rewards


def get_data():
    H_ELBO = get_one_data('H_ELBO_100', ['1-1', '1-1_120', '1-1_229_z', '1-1_382', '1-1_618_z', '1-1_760', '1-1_776',
                                           '1-1_822', '1-1_897', '1-1_921'])
    R_SCL = get_one_data('R-SCL', ['1', '34_h', '41', '42', '395_h', '485', '506', '380_h', '686_h', '906_h'])#'648'
    HCL = get_one_data('HCL', ['-8_55', '-8_77_z', '-8_198', '-8_231_h', '-8_278_h', '-8_298_h', '-8_333_h',
                                         '-8_411', '-8_732_z', '-8_829'])#'-8_730_h'
    pixel = get_one_data('pixel_100', [1, 25, 59, 159, 201, 256, 465, 527, 804, 896])
    return [HCL, H_ELBO, R_SCL, pixel]


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def main():
    plt.figure(figsize=(7.5, 5))
    data = [get_data()[0]]
    label = ['HCL']
    df1 = []
    for i in range(len(data)):
        df1.append(pd.DataFrame(data[i]).melt(var_name='Steps', value_name='Reward'))
        df1[i]['algo'] = label[i]
    df1 = pd.concat(df1)
    df1.index = range(len(df1))
    df1['Steps'] = df1['Steps'] * 10000
    palette1 = sns.color_palette(["#e64c3a"])

    data = get_data()[1:4]
    label = ['H_ELBO', 'R_SCL', 'pixel']
    df2 = []
    for i in range(len(data)):
        df2.append(pd.DataFrame(data[i]).melt(var_name='Steps', value_name='Reward'))
        df2[i]['algo'] = label[i]
    df2 = pd.concat(df2)
    df2.index = range(len(df2))
    df2['Steps'] = df2['Steps'] * 10000
    palette2 = sns.color_palette(["#b18ea0", "#ffa400", "#2dcb70"])

    sns.set_style("white")
    sns.plotting_context("paper")
    ax = sns.lineplot(x="Steps", y="Reward", hue="algo", style="algo", data=df1, palette=palette1, dashes=False, linestyle='--', linewidth=2)
    ax = sns.lineplot(x="Steps", y="Reward", hue="algo", style="algo", data=df2, palette=palette2, dashes=False, linewidth=2)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))

    legend_elements = [
        Line2D([0], [0], color="#e64c3a", label='HCL', linestyle='--'),
        Line2D([0], [0], color="#b18ea0", label='H-ELBO'),
        Line2D([0], [0], color="#ffa400", label='R-SCL'),
        Line2D([0], [0], color="#2dcb70", label='PixelRL'),
    ]
    legend = plt.legend(handles=legend_elements, prop={'size': 15})
    plt.setp(legend.get_lines(), linewidth=2)  # 将所有图例中的线条宽度设为6
    ax.legend_.remove()

    handles, labels = ax.get_legend_handles_labels()
    # plt.title("Cartpole, swingup")
    ax.grid(True, linestyle='-', linewidth=0.5, color='#a3a3a3')
    plt.xlabel("Steps", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    plt.savefig('output.png', dpi=300)
    plt.show()



if __name__ == '__main__':
    main()