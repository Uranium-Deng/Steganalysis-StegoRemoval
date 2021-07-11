import torch
from dataloader import generate_test_data
from train_valid_function import test
from SRNet import SRNet


weight_path = '/home/dengzhirui/0.paper/4.deng/3.SRNet/HUGO_01_Model/125000'

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = SRNet(data_format='NCHW', init_weights=True)
print('model down successfully')

# 数据预处理
data_path = {
    'test_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/cover/',
    'test_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/stego/'
}
batch_size = 1
test_loader = generate_test_data(data_path, batch_size)
print('data_loader down successfully')

test(model=model,
     test_loader=test_loader,
     device=device,
     weight_path=weight_path)


