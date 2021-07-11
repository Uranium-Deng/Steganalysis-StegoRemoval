import torch
import torch.nn as nn
import numpy as np

from dataloader import generate_test_data
from Nets.Generator import Generator

from pylab import plt
import math


# 测试模型在不同MSE范围内图片的数量，主要是为了得到直方图 (在测试集上5000进行)
def get_all_file_MSE(data_path, batch_size=1, flag=0):
    # 实例化模型，加载训练参数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Generator()
    if flag == 0:
        weight_path = '/home/dengruizhi/0.paper/4.deng/2.DDSP/model_save/encoder_model/Model_10500.pth'
    else:
        weight_path = '/home/dengruizhi/0.paper/4.deng/2.DDSP/model_save/GAN_Model/G_Model_10000.pth'
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    print('load weight_file down')
    model.eval()

    # 损失函数
    criterion = nn.MSELoss()
    criterion.to(device)

    data_loader = generate_test_data(data_path, batch_size)

    SD_min = 256
    SD_max = -1
    SD_sum = 0.0
    cnt_li = [0, 0, 0, 0, 0]

    # 这里求解的是result image 和 cover image 之间的MSE
    for index, (images, labels) in enumerate(data_loader):

        labels = labels.to(device)
        model_output = model(images)

        loss = criterion(model_output, labels)
        MSE_value = loss.item()
        SD_value = math.sqrt(MSE_value)
        SD_sum += SD_value

        if SD_value < SD_min:
            SD_min = SD_value
        if SD_value > SD_max:
            SD_max = SD_value
        if SD_value < 5:
            cnt_li[0] += 1
        elif SD_value < 10:
            cnt_li[1] += 1
        elif SD_value < 15:
            cnt_li[2] += 1
        elif SD_value < 20:
            cnt_li[3] += 1
        else:
            cnt_li[4] += 1

        print('MSE: %.6f || SD： %.6f' % (MSE_value, SD_value))

    print('\ncnt_li: ', cnt_li)
    print('SD_max: ', SD_max)
    print('SD_min: ', SD_min)
    print('SD_sum: ', SD_sum)
    print('SD_avg: ', SD_sum / 5000)
    return cnt_li


def generate_histogram(encoder_output, GAN_output):

    # encoder_output 和 GAN_output 均为 get_all_file_MSE 函数的返回值

    # 这两行代码解决 plt 中文显示的问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    # 输入统计数据
    SD_values = ('0-5', '5-10', '10-15', '15-20', '>20')

    bar_width = 0.3  # 条形宽度
    index_male = np.arange(len(SD_values))  # 男生条形图的横坐标
    index_female = index_male + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    encoder_bar = plt.bar(index_male, height=encoder_output, width=bar_width, color='b', label='AutoEncoder')
    GAN_bar = plt.bar(index_female, height=GAN_output, width=bar_width, color='orange', label='GAN')

    for i in encoder_bar:
        height = i.get_height()
        plt.text(i.get_x() + i.get_width() / 2, height, str(height), fontsize=9, va="bottom", ha="center")

    for i in GAN_bar:
        height = i.get_height()
        plt.text(i.get_x() + i.get_width() / 2, height, str(height), fontsize=9, va="bottom", ha="center")

    plt.legend()  # 显示图例
    plt.xticks(index_male + bar_width / 2, SD_values)  # 让横坐标轴刻度显示 SD_values 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('Number of Image')
    plt.xlabel('Pixel standard deviation')  # 纵坐标轴标题
    plt.title('AutoEncoder VS GAN')  # 图形标题

    plt.show()


data_path = {
    'cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/source_data/mytest/cover',
    'stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/6.HUGO_1/mytest/stego'
}
batch_size = 1

# encoder_cnt_li = get_all_file_MSE(data_path=data_path, batch_size=batch_size, flag=0)

# GAN_cnt_li = get_all_file_MSE(data_path=data_path, batch_size=batch_size, flag=1)

encoder_cnt_li = [782, 2716, 817, 297, 388]
GAN_cnt_li = [801, 2799, 795, 235, 371]

generate_histogram(encoder_cnt_li, GAN_cnt_li)

'''
with tf.variable_scope("Discriminator"):
    # 接收从著名画家来的画作
    real_art = tf.placeholder(tf.float32, [None, ART_COMPONENTS], name='real_in')
    D_l0 = tf.layers.dense(real_art, 128, tf.nn.relu, name='l')
    # 经过上一层全连接网络，之后获得概率，所以采用sigmoid函数

    # prob_artist0 是著名画家的画的概率，得到的结果是一个0到1的数字,sigmoid 函数不可能到1呀，满足之后的tf.log() 自变量 x > 0
    prob_artist0 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')

    # 接收从新手画家G的新手画作
    D_l1 = tf.layers.dense(G_out, 128, tf.nn.relu, name='l', reuse=True)
    # 画作是新手画的，但是认为是专业画家画的概率
    prob_artist1 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)


# 这里的prob_artist0和prob_artist1不能简单的认成：专业画家画的画，业余选手画的画
# 应该理解为:
# prob_artist0: 这幅画就是专业画家画的，并且认为是专业画家画的概率
# prob_artist1: 这幅画是业余选手画的，但是被认为是专业画家画的概率


# tf.log()函数以自然数e为底
# 尽量让 判别器认为是著名画家画的画，而减少认为是新手画家画的画
D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1 - prob_artist1))

G_loss = tf.reduce_mean(tf.log(1 - prob_artist1))
'''


