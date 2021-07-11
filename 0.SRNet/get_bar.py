import matplotlib.pyplot as plt
import numpy as np
from pylab import plt


# 这两行代码解决 plt 中文显示的问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 中州(泱泱中华)
# 烟雨江南(乌篷船、邻水)、风啸塞北(地阔天高、大漠孤烟)、雨雪辽东(无际雪林、白山黑水)、征蓬陇西(飘零孤蓬)，壮哉！我大美中华！


# 输入统计数据
SD_values = ('WOW', 'S-UNIWARD', 'HUGO')
encoder_output = [0.941, 0.937, 0.955]
GAN_output = [0.956, 0.946, 0.971]

bar_width = 0.3  # 条形宽度
index_male = np.arange(len(SD_values))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标


# 使用两次 bar 函数画出两组条形图
# encoder_bar = plt.bar(index_male, height=encoder_output, width=bar_width, color='b', label='SRNet')
# GAN_bar = plt.bar(index_female, height=GAN_output, width=bar_width, color='orange', label='CBAM-SRNet')
encoder_bar = plt.bar(index_male, height=encoder_output, width=bar_width, color='b')
GAN_bar = plt.bar(index_female, height=GAN_output, width=bar_width, color='orange')

for i in encoder_bar:
    height = i.get_height()
    plt.text(i.get_x()+i.get_width()/2, height, str(height), fontsize=9, va="bottom", ha="center")

for i in GAN_bar:
    height = i.get_height()
    plt.text(i.get_x()+i.get_width()/2, height, str(height), fontsize=9, va="bottom", ha="center")

plt.legend()  # 显示图例
plt.xticks(index_male + bar_width/2, SD_values)  # 让横坐标轴刻度显示 SD_values 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Accuracy of Model')
plt.xlabel('Embedding Rate')  # 纵坐标轴标题
plt.title('SRNet VS CBAM-SRNet')  # 图形标题

plt.show()


