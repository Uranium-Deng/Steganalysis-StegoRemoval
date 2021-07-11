import torch
import torch.nn as nn
from torch import optim

from train_valid_function import train
from dataloader import generate_data
# from SRNet import SRNet
from SRNet_Attention import SRNet_CBAM as SRNet

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = SRNet(data_format='NCHW', init_weights=True)
print('model down successfully')

# 数据预处理
data_path = {
    'train_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/cover/',
    'train_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/stego/',
    'valid_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/cover/',
    'valid_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/stego/'
}

batch_size = {'train': 2, 'valid': 2}
# batch_size = {'train': 8, 'valid': 20}
train_loader, valid_loader = generate_data(data_path, batch_size)
print('data_loader down successfully')

# 训练参数设置
EPOCHS = 180
write_interval = 40
valid_interval = 50
save_interval = 50
learning_rate = 1e-3

# 损失函数 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

load_path = None

# 开始训练
print('start train')
train(model=model,
      train_loader=train_loader,
      valid_loader=valid_loader,
      EPOCHS=EPOCHS,
      optimizer=optimizer,
      criterion=criterion,
      device=device,
      valid_interval=valid_interval,
      save_interval=save_interval,
      write_interval=write_interval,
      load_path=load_path)

