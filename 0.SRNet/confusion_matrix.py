from tflib.generate_input import gen_validation
from functools import partial
from tflib.train_test_function import get_confusion_matrix
from glob import glob

# 训练集和测试集数据路径
VALID_COVER_DIR = '/home/dengruizhi/0.paper/3.datasets/1.dataset/source_data/validation/cover'
VALID_STEGO_DIR = '/home/dengruizhi/0.paper/3.datasets/1.dataset/6.HUGO_1/validation/stego'

gen_valid = partial(gen_validation, VALID_COVER_DIR, VALID_STEGO_DIR)

weight_path = '/home/dengruizhi/0.paper/4.deng/pre_model/HUGO_1_Model'
data_size = len(glob(VALID_COVER_DIR + '/*')) * 2

get_confusion_matrix(gen_valid, weight_path, data_size, 1)


'''
6个模型的混淆矩阵及准确率如下：

HUGO_1_CBAM: 97%
TT: 470/500, FF: 500/500, TF: 30/500, FT: 0/500 || PosCount: 500, NegCount: 500, correct: 0.9700
HUGO_1 source:  96.1%
TT: 461/500, FF: 500/500, TF: 39/500, FT: 0/500 || PosCount: 500, NegCount: 500, correct: 0.9610

WOW_1_CBAM:  94.6%
TT: 446/500, FF: 500/500, TF: 54/500, FT: 0/500 || PosCount: 500, NegCount: 500, correct: 0.9460
WOW_1_source:  93.4%
TT: 434/500, FF: 500/500, TF: 66/500, FT: 0/500 || PosCount: 500, NegCount: 500, correct: 0.9340

SUNI_1_CBAM:  94.5%
TT: 445/500, FF: 500/500, TF: 55/500, FT: 0/500 || PosCount: 500, NegCount: 500, correct: 0.9450
SUNI_1_source:  88.5%
TT: 385/500, FF: 500/500, TF: 115/500, FT: 0/500 || PosCount: 500, NegCount: 500, correct: 0.8850
'''
