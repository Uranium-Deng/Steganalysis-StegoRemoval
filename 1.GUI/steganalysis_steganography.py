import os
import sys
import numpy as np
import imageio
import tensorflow as tf

from tools.SRNet import SRNet


def insert_info(file_path, method, insert_degree) -> int:
    # method: ['WOW', 'S-UNIWARD', 'HUGO']
    # insert_degree: ['0.4bpp', '0.7bpp', '1bpp']

    print('filename: %s || method: %s || insert_degree: %s' % (file_path.split('/')[-1], method, insert_degree))

    if method == 'HUGO':
        method = 'HUGO_like'
    stego_file_path = '/home/dengruizhi/0.paper/4.deng/1.GUI/2.test_photos/stego'
    insert_degree = float(insert_degree.split('b')[0])

    exec_command = './%s -v -i %s -O %s -a %f'

    try:
        # 生成的含密图像和原始图像的名称相同
        os.system(exec_command % ('tools/' + method, file_path, stego_file_path, insert_degree))
        return 1
    except:
        return 0

# file_path = '/home/dengruizhi/0.paper/4.deng/1.GUI/2.test_photos/cover/1.pgm'
# method = 'WOW'
# insert_degree = '0.7bpp'
#
# insert_info(file_path, method, insert_degree)


def judge_photo(file_path, method, insert_degree):
    tf.reset_default_graph()
    print('file_path: ', file_path)
    print('method: ', method)
    print('insert_degree: ', insert_degree)

    # 这里根据隐写术和嵌入率选择相应的模型对file_path文件进行判断
    model = SRNet(is_training=False, data_format='NCHW')
    print('SRNet model successfully')

    load_path = ''
    if method == 'WOW' and insert_degree == '0.4bpp':
        load_path = '/home/dengruizhi/0.paper/4.deng/8.tf_model/1.source_model/WOW_04_Model'
    elif method == 'WOW' and insert_degree == '0.7bpp':
        load_path = '/home/dengruizhi/0.paper/4.deng/8.tf_model/1.source_model/WOW_07_Model'
    elif method == 'WOW' and insert_degree == '1bpp':
        load_path = '/home/dengruizhi/0.paper/4.deng/8.tf_model/1.source_model/WOW_1_Model'
    elif method == 'S-UNIWARD' and insert_degree == '0.4bpp':
        load_path = '/home/dengruizhi/0.paper/4.deng/8.tf_model/1.source_model/SUNI_04_Model'
    elif method == 'S-UNIWARD' and insert_degree == '0.7bpp':
        load_path = '/home/dengruizhi/0.paper/4.deng/8.tf_model/1.source_model/SUNI_07_Model'
    elif method == 'S-UNIWARD' and insert_degree == '1bpp':
        load_path = '/home/dengruizhi/0.paper/4.deng/8.tf_model/1.source_model/SUNI_1_Model'
    elif method == 'HUGO' and insert_degree == '0.4bpp':
        load_path = '/home/dengruizhi/0.paper/4.deng/8.tf_model/1.source_model/HUGO_04_Model'
    elif method == 'HUGO' and insert_degree == '0.7bpp':
        load_path = '/home/dengruizhi/0.paper/4.deng/8.tf_model/1.source_model/HUGO_07_Model'
    elif method == 'HUGO' and insert_degree == '1bpp':
        load_path = '/home/dengruizhi/0.paper/4.deng/8.tf_model/1.source_model/HUGO_1_Model'

    print('load_path: ', load_path)

    img = imageio.imread(file_path)
    batch = np.empty(shape=[1, img.shape[0], img.shape[1], 1])
    batch[0, :, :, 0] = img
    output_label = model.build_model(batch)

    saver = tf.train.Saver(max_to_keep=10000)

    with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(load_path)
        saver.restore(sess, model_file)
        output_label = sess.run(tf.argmax(output_label, 1))[0]
    return output_label


# file_path = '/home/dengruizhi/0.paper/4.deng/1.GUI/2.test_photos/cover/1.pgm'
# method = 'WOW'
# insert_degree = '0.7bpp'
# judge_photo()

