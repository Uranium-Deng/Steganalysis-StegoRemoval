import torch
import argparse
from torch import optim
import torch.nn as nn

from Nets.Generator import Generator
from Nets.Descriminator import Discriminator
from dataloader import generate_data_loader
from train_GAN_func import train_GAN


parser = argparse.ArgumentParser('DDSP GAN Train')
parser.add_argument('--train_batch_size', default=1, type=int, help='train_batch_size')
parser.add_argument('--valid_batch_size', default=1, type=int, help='valid_batch_size')
parser.add_argument('--n_epochs', default=8, type=int, help='number of epochs')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
parser.add_argument('--train_interval', default=10, type=int, help='train interval')
parser.add_argument('--valid_interval', default=15, type=int, help='validation interval')
parser.add_argument('--save_interval', default=13, type=int, help='save interval')
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# data_path = {
#     'train_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/cover/',
#     'train_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/stego/',
#     'valid_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/cover/',
#     'valid_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/stego/'
# }

data_path = {
    'train_cover': '/media/dengruizhi/BIGDENGU/0.paper/0.copy/1.dataset/source_data/train/cover',
    'train_stego': '/media/dengruizhi/BIGDENGU/0.paper/0.copy/1.dataset/6.HUGO_1/train/stego',
    'valid_cover': '/media/dengruizhi/BIGDENGU/0.paper/0.copy/1.dataset/source_data/validation/cover',
    'valid_stego': '/media/dengruizhi/BIGDENGU/0.paper/0.copy/1.dataset/6.HUGO_1/validation/stego'
}

load_path = '/home/dengruizhi/0.paper/4.deng/2.DDSP/model_save/HUGO_1_Model/Model_20000.pth'

batch_size = {'train': args.train_batch_size, 'valid': args.valid_batch_size}

train_loader, valid_loader = generate_data_loader(data_path, batch_size)


G_model = Generator()
G_optimizer = optim.Adam(G_model.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

D_model = Discriminator()
D_optimizer = optim.Adam(D_model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

criterion = nn.MSELoss()

train_GAN(generator=G_model,
          discriminator=D_model,
          criterion=criterion,
          G_optimizer=G_optimizer,
          D_optimizer=D_optimizer,
          train_loader=train_loader,
          valid_loader=valid_loader,
          device=device,
          train_interval=args.train_interval,
          valid_interval=args.valid_interval,
          save_interval=args.save_interval,
          load_path=load_path,
          n_epochs=args.n_epochs
          )


