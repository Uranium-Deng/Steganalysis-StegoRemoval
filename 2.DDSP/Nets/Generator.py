import torch
import torch.nn as nn

from .utils import ResBlock_Encoder, DownSample_Encoder, MinMax_Norm


class Encoder(nn.Module):
    def __init__(self, in_channels=1, n_residual_blocks=16):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            # encoder的第一层是Min-Max Normalization, 此处舍去 在forward中实现
            # 公式为： x = (x - x_min) / (x_max - x_min), sklearn.preprocess 模块实现
            nn.Conv2d(in_channels, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True),
        )

        self.down_sample_block = DownSample_Encoder(in_channels=64)

        # 16个残差模块
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResBlock_Encoder(in_channels=64))
        self.residual_blocks = nn.Sequential(*res_blocks)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        # print('Encoder input shape: ', x.shape)  # [n, 1, 256, 256]
        x = MinMax_Norm(x)
        # print('encoder min_max_scale: ', x)
        # print('encoder min_max_scale shape: ', x.shape)

        x = self.conv1(x)
        down_sample = self.down_sample_block(x)
        x = self.residual_blocks(down_sample)
        x = self.conv2(x)
        ret = x + down_sample
        # print('Encoder output shape: ', ret.shape)  # [n, 64, 128, 128]
        return ret

# model = Encoder(1, 16)
# print(model)


class Decoder(nn.Module):
    def __init__(self, in_channels=64):
        super(Decoder, self).__init__()
        '''
        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, out_channels=1, kernel_size=9, stride=1, padding=4),
            nn.Tanh(),
            # 最后还需要将图片从[-1, 1] 转换成为 [0, 255], 在forward()中实现
            # 计划先用minmax_scale将数据范围转换到[0, 1] 之后乘以255 转换为[0, 255]
        )
        '''

        # 将上面的tanh()输出的值域[-1, 1]变成[0, 1] 出现了问题，故将文章使用的tanh()函数修改为sigmoid()函数
        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels=1, kernel_size=9, stride=1, padding=4),
            nn.Sigmoid(),
            # 最后还需要将图片从[0, 1] 转换成为 [0, 255], 在forward()中实现
            # 计划先用minmax_scale将数据范围转换到[0, 1] 之后乘以255 转换为[0, 255]
        )

    def forward(self, x):
        # print('Decoder input shape: ', x.shape)  # [n, 64, 128, 128]
        # ret = self.block1(x)
        # ret = MinMax_Norm(ret, require_grad=True).mul(255.0).add(0.5).clamp(0, 255)

        ret = self.block2(x)
        ret = ret.mul(255.0).add(0.5).clamp(0, 255)  # 将sigmoid函数输出值域从[0, 1] -> [0, 255]
        # print('Encoder output shape: ', ret.shape)  # [n, 1, 256, 256]
        return ret


class Generator(nn.Module):
    def __init__(self, init_weights=True):
        super(Generator, self).__init__()
        self.encoder = Encoder(1, 16)
        self.decoder = Decoder(64)

        if init_weights:
            self._init_weights()

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

