import torch
import torch.nn as nn
from sklearn.preprocessing import maxabs_scale, minmax_scale


class ResBlock_Encoder(nn.Module):
    def __init__(self, in_channels=64):
        super(ResBlock_Encoder, self).__init__()
        self.residual_block_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        return x + self.residual_block_encoder(x)


class DownSample_Encoder(nn.Module):
    def __init__(self, in_channels=64):
        super(DownSample_Encoder, self).__init__()
        self.down_sample_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.down_sample_encoder(x)


class ResBlock_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock_Discriminator, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


def MinMax_Norm(x, require_grad=False, batch_size=2):
    # 输入格式 x: [N, C, H, W] 对图片进行min_max标准化

    # x = x.cpu()  # 将向x从cuda中转换到cpu中只是为了得到其形状
    # input_shape = x.shape
    # x = x.reshape(shape=(input_shape[0] * input_shape[1], -1))
    # if require_grad:
    #     # x = torch.from_numpy(minmax_scale(x.t().detach().numpy())).t()
    #     # 禁止使用detach()函数，for循环手动实现minmax Norm标准化
    #     for row in range(x.shape[0]):
    #         temp_min, temp_max = x[row].min(), x[row].max()
    #         x[row] = (x[row] - temp_min) / (temp_max - temp_min)
    # else:
    #     x = torch.from_numpy(minmax_scale(x.t())).t()
    # x = x.reshape(shape=input_shape).float()
    # return x.cuda()

    if require_grad:
        # decoder 最后将cuda中的数据进行标准化
        temp_x = x.cpu()  # 将向x从cuda中转换到cpu中只是为了得到其形状
        input_shape = temp_x.shape
        x = x.reshape(shape=(input_shape[0] * input_shape[1], -1))

        mid_x = x.cpu()  # 将向x从cuda中转换到cpu中只是为了得到其形状
        mid_shape = mid_x.shape

        # 这里的for循环会造成梯度backward 反向传播时，由于inline内联操作出现问题
        for row in range(mid_shape[0]):
            temp_min, temp_max = x[row].min(), x[row].max()
            x[row] = (x[row] - temp_min) / (temp_max - temp_min)

        x = x.reshape(shape=input_shape)

        print('required_grad x: ', x)
        print('required_grad x.shape: ', x.shape)

        return x
    else:
        # encoder 一开始调用min_max_scale进行标准化
        # x = x.cpu()
        input_shape = x.shape
        x = x.reshape(shape=(input_shape[0] * input_shape[1], -1))
        x = torch.from_numpy(minmax_scale(x.t())).t()
        x = x.reshape(shape=input_shape).float()
        return x.cuda()


def MaxAbs_Norm(x):
    # 输入格式 x: [N, C, H, W] 对图片进行max_abs标准化
    # temp_x = x.cpu()
    input_shape = x.shape
    x = x.reshape(shape=(input_shape[0] * input_shape[1], -1))
    x = torch.from_numpy(maxabs_scale(x.t())).t()
    x = x.reshape(shape=input_shape).float()
    return x.cuda()

