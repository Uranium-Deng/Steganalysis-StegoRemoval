import torch
import torch.nn as nn


class Regularization(nn.Module):
    def __init__(self, model, weight_decay, p=2):
        """
        :param model: 模型
        :param weight_decay: 正则化参数
        :param p: 第几范数，默认第二范数
        """
        super(Regularization, self).__init__()
        assert weight_decay > 0, 'weight_decay error, weight_decay should > 0'

        self.model = model
        self.weight_decay = weight_decay
        self.p = p

        self.weight_list = []
        self.reg_loss = 0
        self.device = None

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.get_weights()
        self.regularization_loss()
        return self.reg_loss

    def get_weights(self):
        # 从model中获取所有的权重参数
        for name, param in self.model.named_parameters():
            if 'weight' not in name:
                continue
            temp = (name, param)
            self.weight_list.append(temp)

    def regularization_loss(self):
        # 计算weight的L2正则
        self.reg_loss = 0
        for name, weight in self.weight_list:
            l2_reg = torch.norm(weight, p=self.p)
            self.reg_loss += l2_reg

    def weight_info(self):
        print('-------------------weight information start-------------------')
        for name, param in self.weight_decay:
            print(name)
        print('-------------------weight information down--------------------')


