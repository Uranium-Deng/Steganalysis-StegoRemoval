from glob import glob
from functools import partial
from tflib.train_test_function import test
from Nets.SRNet import SRNet
from tflib.generate_input import gen_validation


TEST_COVER_DIR = '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/cover'
TEST_STEGO_DIR = '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/stego'

test_batch_size = 40
LOG_DIR = 'insert/'  # 模型保存的文件所在路径，注意替换
LOAD_CKPT = LOG_DIR + 'insert.ckpt'  # 指明记载某个ckpt模型文件， 注意替换

test_gen = partial(gen_validation, TEST_COVER_DIR, TEST_STEGO_DIR)

test_ds_size = len(glob(TEST_COVER_DIR + '/*')) * 2
print('test_ds_size', test_ds_size)

if test_ds_size % test_batch_size != 0:
    raise ValueError("change batch size for testing!")

# 开始测试
test(SRNet, test_gen, test_batch_size, test_ds_size, LOAD_CKPT)

