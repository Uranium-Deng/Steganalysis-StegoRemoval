import torch
from os import listdir
from os.path import join
import imageio
from torch.utils.data import Dataset, DataLoader


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
        print(self.filename_list[index], end=' || ')
        cover_img = torch.unsqueeze(torch.tensor(imageio.imread(join(self.cover_path, self.filename_list[index]))), 0)
        stego_img = torch.unsqueeze(torch.tensor(imageio.imread(join(self.stego_path, self.filename_list[index]))), 0)

        return stego_img.float(), cover_img.float()

        # cover_img = imageio.imread(join(self.cover_path, self.filename_list[index]))
        # cover_img = torch.unsqueeze(torch.from_numpy(minmax_scale(cover_img)), dim=0).float()
        #
        # stego_img = imageio.imread(join(self.stego_path, self.filename_list[index]))
        # stego_img = torch.unsqueeze(torch.from_numpy(minmax_scale(stego_img)), dim=0).float()
        #
        # return stego_img, cover_img


def generate_data_loader(data_path, batch_size):
    train_data = DataGenerator(data_path['train_cover'], data_path['train_stego'])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size['train'], shuffle=True, num_workers=0, drop_last=True)

    valid_data = DataGenerator(data_path['valid_cover'], data_path['valid_stego'])
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size['valid'], drop_last=True)

    return train_loader, valid_loader


def generate_test_data(data_path, batch_size=1):
    # 要得到每一张图片的MSE，所以batch_size等于1
    test_data = DataGenerator(data_path['cover'], data_path['stego'])
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=0)
    return test_loader


# data_path = {
#     'train_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/cover/',
#     'train_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/stego/',
#     'valid_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/cover/',
#     'valid_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/stego/'
# }
# batch_size = {'train': 1, 'valid': 1}
#
# train_loader, valid_loader = generate_data_loader(data_path, batch_size)
#
#
# for index, (images, labels) in enumerate(train_loader):
#     print('images.shape: ', images.shape)
#     print('labels.shape: ', labels.shape)
#
#     for i in range(images.shape[0]):
#         plt.subplot(1, 2, 1)
#         plt.title('stego_img_{}'.format(i))
#         plt.imshow(images[i][0], cmap='gray')
#
#         plt.subplot(1, 2, 2)
#         plt.title('cover_img_{}'.format(i))
#         plt.imshow(labels[i][0], cmap='gray')
#
#         plt.show()
#     break
