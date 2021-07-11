import os
from functools import partial
from glob import glob
from Nets.SRNet import SRNet as SRNet
# from Nets.SRNet_Attention import SRNet_CBAM as SRNet
from Nets.Optimizer import AdamaxOptimizer
from tflib.train_test_function import train
from tflib.generate_input import gen_flip_and_rot, gen_validation

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用1号显卡


# 超参数设置
train_batch_size = 16  # 训练时的batch_size
valid_batch_size = 40  # valid时的batch_size, 所以valid上每一个epoch有25次测试
train_interval = 100  # 每隔100个iterations，保存一次训练的loss和accuracy, 共5k次
valid_interval = 5000  # 每隔5000个iterations在validation set上测试一次，计算平均正确率，测试100次数
max_iter = 500000  # 训练过程一共500k个iterations
save_interval = 5000  # 每隔5000个iterations保存一次ckpt模型文件，保存100次
num_runner_threads = 10  # 训练时向队列中放入数据的线程数量


# 训练集和测试集数据路径
TRAIN_COVER_DIR = '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/cover'
TRAIN_STEGO_DIR = '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/stego'
VALID_COVER_DIR = '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/cover'
VALID_STEGO_DIR = '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/stego'

gen_train = partial(gen_flip_and_rot, TRAIN_COVER_DIR, TRAIN_STEGO_DIR)
gen_valid = partial(gen_validation, VALID_COVER_DIR, VALID_STEGO_DIR)

# 模型保存路径
LOG_DIR = '/home/dengruizhi/0.paper/4.deng/WOW_04_Model'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 加载的模型的路径
load_path = '/home/dengruizhi/0.paper/4.deng/logfiles/WOW_07_Model'

train_ds_size = len(glob(TRAIN_COVER_DIR + '/*')) * 2
valid_ds_size = len(glob(VALID_COVER_DIR + '/*')) * 2
print('train_ds_size: ', train_ds_size)
print('valid_ds_size: ', valid_ds_size)

if valid_ds_size % valid_batch_size != 0:
    raise ValueError('valid_ds_size % valid_batch_size != 0, change batch size for validation')

optimizer = AdamaxOptimizer
boundaries = [400000]  # 在第400k的时候学习率改变
values = [0.001, 0.0001]

# 调用训练函数开始训练
train(SRNet, gen_train, gen_valid, train_batch_size, valid_batch_size, train_ds_size,
      valid_ds_size, optimizer, boundaries, values, train_interval, valid_interval,
      max_iter, save_interval, LOG_DIR, num_runner_threads, load_path)



