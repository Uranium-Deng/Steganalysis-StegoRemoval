import imageio
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from os.path import join
import os
import random

from Nets.Generator import Generator
from utils.get_BER import generate_file_BER


def get_psnr(source_path, stego_path):
    # 计算两幅图像的PSNR值
    source_img = imageio.imread(source_path)
    stego_img = imageio.imread(stego_path)

    mse = np.mean((source_img/1.0 - stego_img/1.0) ** 2)
    if mse < 1.0e-10:
        return 100.0
    return 10 * math.log10(255.0**2 / mse)

# PSNR > 40 说明图像质量很好，和原始图片很接近
# PSNR在30-40之间，图像质量较好，察觉到失真但是可以接收
# PSNR在20-30之间，图像质量差
# PSNR在20一下，无法接受


def get_residual(cover_file, stego_file):
    # 显示原始图像，变换后的图像，残差图像
    cover_file = plt.imread(cover_file)
    stego_file = plt.imread(stego_file)
    absres = np.abs(cover_file - stego_file)

    # print('cover_file shape: ', cover_file.shape)
    # print('stego_file shape: ', stego_file.shape)

    plt.subplot(1, 3, 1)
    plt.imshow(cover_file, cmap='gray')
    plt.title('stego image')
    plt.subplot(1, 3, 2)
    plt.imshow(stego_file, cmap='gray')
    plt.title('model_output image')
    plt.subplot(1, 3, 3)
    plt.imshow(absres, cmap='gray')
    plt.title('residual image')
    plt.show()


def get_mean_square_error(x, y):
    # 返回平均每一个像素点差值的平方
    ans = 0.0
    input_shape = x.shape
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            ans += (x[i][j] - y[i][j])**2
    return ans / (input_shape[0] * input_shape[1])


# 通过像素点的像素值计算误码率,方法错误
# def get_BER(x, y):
#     ans_cnt = 0
#     input_shape = x.shape
#     for i in range(input_shape[0]):
#         for j in range(input_shape[1]):
#             if x[i][j] != y[i][j]:
#                 ans_cnt += 1
#     print('ans_cnt: ', ans_cnt)
#     ans_BER = ans_cnt / (input_shape[0] * input_shape[1])
#     return ans_BER


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1_path, img2_path):
    # 计算两图像的SSIM值
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def test_model(stego_file):
    # 传入图片, 得到并保存模型输出图片, 计算输入图片和模型输出图片的PSNR SSIM以及MSE

    # 对图片增加两个维度
    raw_data = imageio.imread(stego_file)
    stego_img = torch.unsqueeze(torch.unsqueeze(torch.tensor(raw_data), 0), 0).float()

    # 两个权重文件
    encoder_weight_path = '/home/dengruizhi/0.paper/4.deng/2.DDSP/model_save/encoder_model/Model_10500.pth'
    GAN_weight_path = '/home/dengruizhi/0.paper/4.deng/2.DDSP/model_save/GAN_Model/G_Model_10000.pth'

    # 模型准备
    model = Generator()
    model.load_state_dict(torch.load(GAN_weight_path))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 获得模型输出
    model_output = model(stego_img)
    model_output = model_output.detach().cpu().numpy()
    ans_img = model_output[0][0].astype(int)

    # print(type(ans_img))
    # print('max value: ', ans_img.max())
    # print('min value: ', ans_img.min())
    # print(ans_img)

    # 保存模型输出图片
    img_name = stego_file.split('/')[-1]
    save_dir = '/home/dengruizhi/0.paper/4.deng/2.DDSP/test_img/all_index_imgs'
    new_path = join(save_dir, img_name)
    imageio.imsave(new_path, ans_img)

    # 计算PSNR
    psnr = get_psnr(stego_file, new_path)
    print('PSNR: ', psnr)

    # 计算SSIM
    ssim = calculate_ssim(stego_file, new_path)
    print('SSIM: ', ssim)

    # 计算MSE
    mse = get_mean_square_error(raw_data, ans_img)
    SD = math.sqrt(mse)
    print('MSE: ', mse)
    print('SD: ', SD)

    # 计算BER
    BER = generate_file_BER(stego_file, new_path)
    print('BER: ', BER)

    # 显示含密图像, 修改后的图像, 两者的残差图像
    # get_residual(stego_file, new_path)

    return [psnr, ssim, mse, SD, BER]


def get_table(test_data_path):
    # 生成图像和含密图像
    cover_data_path = '/home/dengruizhi/0.paper/3.datasets/1.dataset/source_data/mytest/cover'
    filenames = os.listdir(test_data_path)
    # num_file = len(filenames)
    choiced_img = random.sample(filenames, 10)
    print('choiced_img: ', choiced_img)
    final_ans = []
    avg_ans = []
    for filename in choiced_img:
        stego_file = join(test_data_path, filename)
        tmp_ans = test_model(stego_file)
        final_ans.append(tmp_ans)
    print('\n\nfinal_ans: ', final_ans)

    for i in range(5):
        tmp = [j[i] for j in final_ans]
        avg_ans.append(sum(tmp) / 10)

    # 每一评价指标的平均值
    print('avg_ans: ', avg_ans)
    print('down')

# stego_path = '/home/dengruizhi/0.paper/4.deng/2.DDSP/test_img/stego_img/19547.pgm'
# test_model(stego_path)

test_data = '/home/dengruizhi/0.paper/3.datasets/1.dataset/6.HUGO_1/mytest/stego'
get_table(test_data)


'''
输出测试:
17836.pgm (舍弃)
encoder
PSNR:  34.51604993163772
SSIM:  0.9768665309245567
MSE:  23.880508422851562
GAN
PSNR:  34.190084980424416
SSIM:  0.9752261457217731
MSE:  24.710433959960938
-----------------------------------
18427.pgm (可以考虑)
encoder
PSNR:  34.305954592822864
SSIM:  0.9803280469853379
MSE:  28.044937133789062
GAN
PSNR:  35.25347090575521
SSIM:  0.9813807201071596
MSE:  27.5728759765625
-----------------------------------
18561.pgm (可以考虑)
encoder
PSNR:  31.201379884105677
SSIM:  0.9758460713511701
MSE:  20.918609619140625
GAN
PSNR:  33.040052232335114
SSIM:  0.9809484878409438
MSE:  16.219207763671875
-----------------------------------
17236.pgm 
encoder
PSNR:  31.26389362529768
SSIM:  0.9650216177229564
MSE:  27.19647216796875
GAN
PSNR:  31.524430902060274
SSIM:  0.9653464912000138
MSE:  21.370285034179688
-----------------------------------
19547.pgm (舍弃)
encoder
PSNR:  35.316969200970554
SSIM:  0.9860602198842571
MSE:  10.7078857421875
GAN
PSNR:  33.32176697102837
SSIM:  0.9815994459719656
MSE:  8.128875732421875
----------------------------------------------------------------------
avg_ans:  [0.8668212890625, 158.3248062133789, 10.348574371334383, 29.10681081625011, 0.9677427386353072]
avg_ans:  [0.9423309326171875, 162.43157043457032, 10.731899608238397, 27.58141098601241, 0.9622646888274439]
avg_ans:  [0.9169540405273438, 405.40599212646487, 12.279803210752807, 28.392867651885723, 0.9530453665122376]
avg_ans:  [0.94100341796875, 97.25937042236328, 9.106852720887876, 29.21645648873399, 0.9701477594762494]
avg_ans:  [0.8642868041992188, 108.52584381103516, 8.48199981720024, 29.53564091468845, 0.9691432474455917]
avg_ans:  [0.9188140869140625, 224.88195037841797, 12.109032282297651, 28.402793860373045, 0.9680685084588649]
avg_ans:  [0.8779861450195312, 348.96684112548826, 12.726392403718465, 29.28760549373979, 0.9554270524957102]
avg_ans:  [0.8755630493164063, 54.454551696777344, 6.19807633067986, 31.647809709387285, 0.9692298123504711] 好
avg_ans:  [0.9309539794921875, 83.54339752197265, 8.403642717762304, 30.192902198709664, 0.9758577514070744]
avg_ans:  [0.8831008911132813, 53.14381713867188, 6.945343839739825, 30.756730863588132, 0.966574358825941]
avg_ans:  [0.93687744140625, 150.76854858398437, 9.456491899912477, 30.060236471763403, 0.9689824361255225]
avg_ans:  [0.9112823486328125, 76.61582946777344, 8.07873442609292, 30.66048893738274, 0.9704910817077843]
avg_ans:  [0.8786880493164062, 77.54087677001954, 7.794392344903474, 30.91922508301054, 0.9760555578220551]
avg_ans:  [0.923358154296875, 44.31521911621094, 6.322859066488261, 31.115257462104125, 0.9726550927440576]
avg_ans:  [0.9050155639648437, 52.998948669433595, 6.992206586721847, 31.29948902421619, 0.9711246784422352]
avg_ans:  [0.8629562377929687, 65.93675537109375, 7.417879732048941, 30.138234259025808, 0.9670584828320538]
'''

