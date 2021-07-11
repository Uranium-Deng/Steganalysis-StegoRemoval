#!/usr/bin/env python
# coding=utf-8

import os
from os.path import join
import imageio

# 计算图像的误码率 bit error rate


def generate_BER(x: int, y: int):
    # 计算图像两个向素值相同比特的个数
    x_li, y_li = list(bin(x))[2:], list(bin(y))[2:]

    for i in range(8 - len(x_li)):
        x_li.insert(0, '0')
    for i in range(8 - len(y_li)):
        y_li.insert(0, '0')

    same_bit = 0
    for i in range(8):
        if x_li[i] == y_li[i]:
            same_bit += 1
    return same_bit


def generate_file_BER(source_file: str, target_file: str):
    # 计算每一张图片的BER: bit error rate 误码率
    source_file = imageio.imread(source_file)
    target_file = imageio.imread(target_file)

    source_shape, target_shape = source_file.shape, target_file.shape
    assert source_shape == target_shape, 'source file shape is different from target file shape'

    total_same_bits = 0
    total_bits = source_shape[0] * source_shape[1] * 8.0
    for row in range(source_shape[0]):
        for col in range(source_shape[1]):
            total_same_bits += generate_BER(source_file[row][col], target_file[row][col])
    return total_same_bits / total_bits


def generate_test_data():
    print('start successfully')
    result_base_path = '/home/dengruizhi/0.paper/4.deng/2.DDSP/test_img/random_10'
    result_dir_names = ['8.demo', '9.demo', '10.demo']
    source_base_path = '/home/dengruizhi/0.paper/3.datasets/1.dataset/6.HUGO_1/mytest/stego'

    for dir_name in result_dir_names:
        dir_path = join(result_base_path, dir_name)
        result_files = os.listdir(dir_path)

        BER_per_dir = []

        for file in result_files:
            if file.endswith('.txt'):
                continue
            result_file = join(dir_path, file)
            stego_file = join(source_base_path, file)
            BER_per_dir.append(generate_file_BER(stego_file, result_file))
        print('{}: '.format(dir_name), BER_per_dir)
        print('average BER: ', sum(BER_per_dir) / 10.0)

    print('down successfully')


generate_test_data()


'''
论文中使用的三个BER
8.demo average BER:  0.6898359298706055
9.demo average BER:  0.6793571472167969
10.demo average BER:  0.6955537796020508
'''

