import torch
import argparse
from torch import optim
import torch.nn as nn

from Nets.Generator import Generator
from dataloader import generate_data_loader
from train_test_autoencoder import train


parser = argparse.ArgumentParser('DDSP Train')
parser.add_argument('--train_batch_size', default=14, type=int, help='train_batch_size')
parser.add_argument('--valid_batch_size', default=14, type=int, help='valid_batch_size')
parser.add_argument('--n_epochs', default=15, type=int, help='number of epochs')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
parser.add_argument('--train_interval', default=100, type=int, help='train interval')
parser.add_argument('--valid_interval', default=500, type=int, help='validation interval')
parser.add_argument('--save_interval', default=500, type=int, help='test interval')
args = parser.parse_args()

# 训练集14000张图片，batch_size： 14,一次epoch需要1000次iterations
# 每一个epoch，测试和保存模型各2次，记录loss10次


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path = {
    'train_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/cover/',
    'train_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/stego/',
    'valid_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/cover/',
    'valid_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/stego/'
}
batch_size = {'train': args.train_batch_size, 'valid': args.valid_batch_size}

train_loader, valid_loader = generate_data_loader(data_path, batch_size)

load_path = None

model = Generator()
# print('model: ', model)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
criterion = nn.MSELoss()

train(model=model,
      train_loader=train_loader,
      valid_loader=valid_loader,
      n_epochs=args.n_epochs,
      device=device,
      optimizer=optimizer,
      criterion=criterion,
      train_interval=args.train_interval,
      valid_interval=args.valid_interval,
      save_interval=args.save_interval,
      load_path=load_path,
      )
