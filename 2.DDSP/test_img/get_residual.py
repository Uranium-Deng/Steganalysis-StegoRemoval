#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

cover_file_path = '/home/dengruizhi/0.paper/3.datasets/1.dataset/source_data/mytest/cover/18561.pgm'
stego_file_path = './stego_img/18561.pgm'
result_file_path = './result_img/18561.pgm'

cover_file = plt.imread(cover_file_path)
stego_file = plt.imread(stego_file_path)
result_file = plt.imread(result_file_path)

res_file1 = np.abs(cover_file - result_file)
res_file2 = np.abs(stego_file - result_file)


plt.subplot(1, 3, 1)
plt.imshow(cover_file, cmap='gray')
plt.title('cover image')
plt.subplot(1, 3, 2)
plt.imshow(stego_file, cmap='gray')
plt.title('stego image')
plt.subplot(1, 3, 3)
plt.imshow(result_file, cmap='gray')
plt.title('generated image')
plt.savefig('cover_stego_result_18561.png')

plt.show()




