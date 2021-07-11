from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio
import random
from os import listdir
from os.path import join
import torch


class DataGenerator(Dataset):
    def __init__(self, cover_dir, stego_dir):
        self.cover_path = cover_dir
        self.stego_path = stego_dir

        cover_list = listdir(cover_dir)
        stego_list = listdir(stego_dir)
        self.filename_list = cover_list

        cover_len = len(cover_list)
        stego_len = len(stego_list)
        assert cover_len != 0, "the cover directory:{} is empty!".format(cover_dir)
        assert stego_len != 0, "the stego directory:{} is empty!".format(stego_dir)
        assert cover_len == stego_len, "the cover directory and stego directory don't have the same number files, " \
                                       "respectively： %d, %d" % (cover_len, stego_len)

        img = imageio.imread(join(self.cover_path, self.filename_list[0]))
        self.img_shape = img.shape

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        batch = np.empty(shape=(2, self.img_shape[0], self.img_shape[1], 1), dtype='uint8')

        batch[0, :, :, 0] = imageio.imread(join(self.cover_path, self.filename_list[index]))
        batch[1, :, :, 0] = imageio.imread(join(self.stego_path, self.filename_list[index]))

        label = torch.tensor([0, 1], dtype=torch.int64)

        rot = random.randint(0, 3)
        if random.random() < 0.5:
            return [torch.from_numpy(np.rot90(batch, rot, axes=[1, 2]).copy()), label]
        else:
            return [torch.from_numpy(np.flip(np.rot90(batch, rot, axes=[1, 2]).copy(), axis=2).copy()), label]


def generate_data(data_path, batch_size):
    train_data = DataGenerator(data_path['train_cover'], data_path['train_stego'])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size['train'], shuffle=True, num_workers=2, drop_last=True)

    valid_data = DataGenerator(data_path['valid_cover'], data_path['valid_stego'])
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size['valid'], drop_last=True)

    return train_loader, valid_loader


def generate_test_data(data_path, batch_size):
    mytest_data = DataGenerator(data_path['test_cover'], data_path['test_stego'])
    valid_loader = DataLoader(dataset=mytest_data, batch_size=batch_size, drop_last=True)

    return valid_loader


# data_path = {
#     'train_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/cover/',
#     'train_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/stego/',
#     'valid_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/cover/',
#     'valid_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/stego/'
# }
# batch_size = {'train': 8, 'valid': 8}
#
# train, valid = generate_data(data_path, batch_size)
#
# data_path = {
#     'test_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/cover/',
#     'test_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/stego/'
# }
# batch_size = 1
# test_data = generate_test_data(data_path, batch_size)
#
# for index, (images, labels) in enumerate(test_data):
#     print(images.shape)
#     print(labels.shape)
#     print(labels)
#     images = images.view(-1, 256, 256, 1)
#     labels = torch.squeeze(labels.view(-1, 1))  # 在网络中进行了NCHW转换
#     print(images.shape)
#     print(labels.shape)
#     print(labels)
#     break


