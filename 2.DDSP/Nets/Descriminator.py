import torch
import torch.nn as nn

from .utils import ResBlock_Discriminator, MaxAbs_Norm


# maxabs_scale() 具体实现公式为 x / max(|x|) 最终结果在[-1, 1]之间

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, required_grad=False):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        residual_block_params = [
            [512, 512, 2], [512, 256, 1], [256, 256, 2],
            [256, 128, 1], [128, 128, 2], [128, 64, 1], [64, 64, 2]
        ]
        residual_blocks_list = []
        for param in residual_block_params:
            residual_blocks_list.append(ResBlock_Discriminator(*param))
        self.residual_blocks = nn.Sequential(*residual_blocks_list)

        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        input_shape = x.shape
        # print('discriminator input shape: ', input_shape)
        x = MaxAbs_Norm(x)
        # print('after maxabs_scale x:', x)
        # print('x.shape: ', x.shape)
        x = self.conv1(x)
        x = self.residual_blocks(x)
        x = x.reshape(shape=(input_shape[0], -1))  # flatten 转换为[n, -1] 每一行就是一个feature map
        x = self.classifier(x)
        # print('discriminator output shape: ', x.shape)
        return x


# model = Discriminator(in_channels=1)
# print(model)

