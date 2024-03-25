import numpy as np
from scipy.stats import norm
import torch
import os 
import matplotlib.pyplot as plt
import seaborn as sns
palette = 'colorblind'
colors = ['gold', 'grey', '#d7191c', '#2b83ba' ]
colors = sns.color_palette("muted", n_colors=10)

class STEPD:
    def __init__(self, new_window_size, alpha_w=0.05, alpha_d=0.003):
        self.new_window_size = new_window_size
        self.alpha_w = alpha_w
        self.alpha_d = alpha_d
        self.cnt, self.shift_cnt = 1, 0
        self.data = []
        self.data_visualize = None

    def add_data(self, error_rate , x):
        self.cnt += 1
        if self.data_visualize is None:
            self.data_visualize = x.cpu().detach()[0]
        else:
            self.data_visualize = torch.cat([self.data_visualize, x.cpu().detach()[0,-1,:].unsqueeze(0)], dim=0)
        # if len(self.data) > self.new_window_size and self.is_outlier(error_rate) :
        #     # 如果是异常值，不将其添加到数据中
        #     return
        self.data.append(error_rate)

    def reset(self):
        self.shift_cnt += 1
        self.data = []
        self.cnt = 0
        self.data_visualize = None
    
    def is_outlier(self, value, threshold=3.0):
        # 使用标准差的方法检测异常值，也许这里不需要normalize，绝对大小就能说明问题
        mean_value = np.mean(self.data[-self.new_window_size:])
        std_dev_value = np.std(self.data[-self.new_window_size:])
        z_score = (value - mean_value) / (std_dev_value + 1e-4)
        warning_threshold = norm.ppf(1 - self.alpha_w / 2)
        return z_score > warning_threshold
    
    def run_test(self,):
        if len(self.data) < self.new_window_size:
            # Not enough data for comparison
            return 0, None

        # Extract the most recent time window and the overall time window
        recent_window = self.data[-self.new_window_size:]
        overall_window = self.data

        # Calculate the test statistic
        mean_recent = np.mean(recent_window)
        mean_overall = np.mean(overall_window)
        std_dev_overall = np.std(overall_window)
        n = len(self.data)
        theta_stepd = (mean_recent - mean_overall) / (std_dev_overall / np.sqrt(n))

        # Calculate the warning and drift thresholds
        warning_threshold = norm.ppf(1 - self.alpha_w / 2)
        drift_threshold = norm.ppf(1 - self.alpha_d / 2)

        # Check for warning or drift
        if theta_stepd > drift_threshold:
            # self.plt_distribution(self.data, c1 = colors[0], c2 = colors[1])
            # self.plt_distribution(self.data_visualize[:,0].numpy(), name='value', c1 = colors[2], c2 = colors[3])
            return 1, 3e-3
        else:
            lr = 1e-4 + (3e-3 - 1e-4) * (theta_stepd / drift_threshold)
            return 0, max(lr, 1e-4)

    def plt_distribution(self, data, name='error', c1 = colors[0], c2 = colors[1]):
        sns.set(style="whitegrid", font_scale=1.8)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.figure(figsize=(12, 8))
        plt.plot(data, label='Entire Dataset', color=c1)

        # Highlight the last 100 time steps
        plt.plot(range(len(data) - self.new_window_size, len(data)), data[-self.new_window_size:], label=f'Last Window', color=c2, linewidth=2)
        
        # Plot the mean of the entire dataset
        mean_entire = np.mean(data)
        plt.axhline(mean_entire, color=c1, linestyle='dashed', label=f'Mean (Entire): {mean_entire:.2f}')

        # Plot the mean of the last 100 steps
        mean_last_100 = np.mean(data[-self.new_window_size:])
        plt.axhline(mean_last_100, color=c2, linestyle='dashed', label=f'Mean (Last Window): {mean_last_100:.2f}')

        if name == 'value':
            plt.legend([])
        else:
            plt.legend()
        plt.xlabel('Time Steps')
        plt.ylabel(name)
        # Add legend
        
        # Save the plot as a PDF file
        if not os.path.exists('imgs/drift/'):
            os.mkdir('imgs/drift/')
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        plt.savefig(f'imgs/drift/{name}_plot_{self.shift_cnt}.pdf')
        plt.close()