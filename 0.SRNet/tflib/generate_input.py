import numpy as np
from glob import glob
import imageio
from random import shuffle
import random


# 注意，下面的两个函数，而是generator，因为最后采用的不是return而是yield
def gen_flip_and_rot(cover_dir, stego_dir, thread_idx, num_threads):
    """
    :param cover_dir: cover images目录所在的路径
    :param stego_dir: stego images目录所在的路径
    :param thread_idx: 线程id
    :param num_threads: 线程数量
    :return: 返回翻转或水平镜像后的图片和label [(2, height, width, 1), (0, 1)], 供训练使用
    """
    cover_list = sorted(glob(cover_dir + '/*'))
    stego_list = sorted(glob(stego_dir + '/*'))

    cover_len = len(cover_list)
    stego_len = len(stego_list)
    assert cover_len != 0, "the cover directory:{} is empty!".format(cover_dir)
    assert stego_len != 0, "the stego directory:{} is empty!".format(stego_dir)
    assert cover_len == stego_len, "the cover directory and stego directory don't have the same number files, " \
                                   "respectively： %d, %d" % (cover_len, stego_len)

    # print('cover_len: ', cover_len)
    # print('stego_len: ', stego_len)
    img = imageio.imread(cover_list[0])
    img_shape = img.shape
    batch = np.empty(shape=(2, img_shape[0], img_shape[1], 1), dtype='uint8')

    iterable = list(zip(cover_list, stego_list))
    label = np.array([0, 1], dtype='uint8')

    while True:
        shuffle(iterable)
        for cover_path, stego_path in iterable:
            batch[0, :, :, 0] = imageio.imread(cover_path)
            batch[1, :, :, 0] = imageio.imread(stego_path)

            rot = random.randint(0, 3)
            if random.random() < 0.5:
                yield [np.rot90(batch, rot, axes=[1, 2]), label]
            else:
                yield [np.flip(np.rot90(batch, rot, axes=[1, 2]), axis=2), label]


def gen_validation(cover_dir, stego_dir, thread_idx, num_threads):
    """
    :param cover_dir: cover images目录所在的路径
    :param stego_dir: stego images目录所在的路径
    :param thread_idx: 线程id
    :param num_threads: 线程数量
    :return: 返回图片和label [(2, height, width, 1), (0, 1)], 供测试使用
    """
    cover_list = sorted(glob(cover_dir + '/*'))
    stego_list = sorted(glob(stego_dir + '/*'))

    cover_len = len(cover_list)
    stego_len = len(stego_list)
    assert cover_len != 0, "the cover directory:{} is empty".format(cover_dir)
    assert stego_len != 0, "the stego directory:{} is empty".format(stego_dir)
    assert cover_len == stego_len, "the cover directory and stego directory don't have the same number files, " \
                                   "respectively： %d, %d" % (cover_len, stego_len)

    img = imageio.imread(cover_list[0])
    img_shape = img.shape
    batch = np.empty(shape=(2, img_shape[0], img_shape[1], 1), dtype='uint8')
    label = np.array([0, 1], dtype='uint8')

    iterable = list(zip(cover_list, stego_list))
    while True:
        for cover_path, stego_path in iterable:
            batch[0, :, :, 0] = imageio.imread(cover_path)
            batch[1, :, :, 0] = imageio.imread(stego_path)
            yield [batch, label]




