#!/usr/bin/env python
# coding=utf-8

from PIL import Image
import os

def resize_image(source_dir, target_dir, new_width, new_height):
    print('source dir: ', source_dir)
    print('target dir: ', target_dir)

    source_list = os.listdir(source_dir)
    print('source dir has %d files' % len(source_list))

    cnt = 0
    for filename in source_list:
        img = Image.open(source_dir + filename)
        resize_img = img.resize((new_width, new_height))
        resize_img.save(target_dir + filename)

        cnt += 1
        if cnt % 200 == 0:
            print('cnt: %d down' % cnt)


source_dir = '/home/dengruizhi/0.paper/4.deng/1.GUI/2.test_photos/cover/'
target_dir = '/home/dengruizhi/0.paper/4.deng/1.GUI/2.test_photos/stego/'

new_width = 256
new_height = 256

resize_image(source_dir, target_dir, new_width, new_height)
        




