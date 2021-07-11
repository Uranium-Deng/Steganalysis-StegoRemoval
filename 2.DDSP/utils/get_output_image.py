import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import imageio
import math

from dataloader import generate_test_data
from Nets.Generator import Generator

##################################
# 将autoencoder 的输出以图片的形式保存
##################################


# def save_image(stego_file, model, device, weight_path):
#     stego_img = torch.unsqueeze(torch.unsqueeze(torch.tensor(imageio.imread(stego_file)), 0), 0).float()
#     model.load_state_dict(torch.load(weight_path))
#     model.to(device)
#     model.eval()
#     model_output = model(stego_img)
#     model_output = model_output.detach().cpu().numpy()
#     ans_img = model_output[0][0]
#     # print(ans_img)
#     new_path = '/home/dengruizhi/0.paper/4.deng/2.DDSP/test_img/result_img/1.pgm'
#     imageio.imsave(new_path, ans_img)
#     print('down')


def generate_output_imgae(data_path, batch_size=1, flag=0):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Generator()
    if flag == 0:
        weight_path = '/home/dengruizhi/0.paper/4.deng/2.DDSP/model_save/encoder_model/Model_10500.pth'
    else:
        weight_path = '/home/dengruizhi/0.paper/4.deng/2.DDSP/model_save/GAN_Model/G_Model_10000.pth'
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    print('load weight_file down')
    model.eval()

    # 损失函数
    criterion = nn.MSELoss()
    criterion.to(device)
    data_loader = generate_test_data(data_path, batch_size)

    for index, (images, labels) in enumerate(data_loader):
        if index >= 10:
            break
        labels = labels.to(device)
        model_output = model(images)

        loss = criterion(model_output, labels)
        print('loss_value: ', loss.item())

        model_output = model_output.detach().cpu().numpy()
        images, labels = images.cpu(), labels.cpu()

        for i in range(images.shape[0]):
            plt.subplot(1, 3, 1)
            plt.title('cover_img_{}'.format(index))
            plt.imshow(labels[i][0], cmap='gray')

            plt.subplot(1, 3, 2)
            plt.title('stego_img_{}'.format(index))
            plt.imshow(images[i][0], cmap='gray')

            plt.subplot(1, 3, 3)
            plt.title('output_img_{}'.format(index))
            plt.imshow(model_output[i][0], cmap='gray')

            plt.show()


data_path = {
    'cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/source_data/mytest/cover',
    'stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/6.HUGO_1/mytest/stego'
}
batch_size = 1
generate_output_imgae(data_path, batch_size, flag=0)


# 可视化中间层输出的feature map 以及卷积核的值

# 要输出网络某一层的输出，就是在forward()函数返回的内容，所以只要是修改forward()函数的值
# 其实就是逐层的执行网络中的那些层，只是遇到想要输出的层时，将该输出保存起来，最终返回

# 输出每一层卷积核(一层卷积有多个卷积核)的值，主要用到model.state_dict()函数，函数返回的是一个字典
# key是包含参数的层的weight和bias (如Conv Linear层)，然后根据key取的对应的值
# 通常对卷积核中的值，取最大最小值，然后n等分，画出直方图，观察它的值满足什么样子的一个分布(通常是在正态分布)

# 我们知道卷积和全连接都有weight和bias这两个参数，pooling, activate function 没有参数优化
# BN 也有参数，而且还有四个，分别对应：伽马，贝塔，均值，方差，关于BN还是要好好看看

# Attention 机制
# Squeeze-and_excitation Networks SE Net 引入了通道注意力机制

