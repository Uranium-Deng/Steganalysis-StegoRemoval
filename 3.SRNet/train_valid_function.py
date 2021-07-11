import torch
from tensorboardX import SummaryWriter
import random
from L2_Regularization import Regularization


def adjust_learning_rate(optimizer, gamma, global_step):
    lr = 1e-3 * (gamma ** global_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validation(model, valid_loader, device, criterion, writer, train_step):
    cnt = 0  # validation中所有图片的数量
    total_acc = 0  # 所有图片的accuracy的和
    total_loss = 0  # 所有图片的loss的和
    model.to(device)

    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.view(-1, 256, 256, 1).to(device)
            labels = labels.view(-1, 1).to(device)
            labels = torch.squeeze(labels)

            cnt += images.shape[0]

            model_output = model(images)

            loss = criterion(model_output, labels)
            total_loss += loss.item()
            # 计算accuracy
            model_label = torch.max(model_output, dim=1)[1]
            temp_acc = torch.eq(model_label, labels).sum().item()
            total_acc += temp_acc

        avg_acc = total_acc / cnt
        avg_loss = total_loss / cnt
        writer.add_scalar('valid_accuracy', avg_acc, train_step)
        writer.add_scalar('valid_loss', avg_loss, train_step)
        print('currIter: %d || valid_loss: %.6f || valid_acc: %.6f' % (train_step, avg_loss, avg_acc))


def train(model,
          train_loader,
          valid_loader,
          optimizer,
          criterion,
          device,
          EPOCHS,
          valid_interval=5000,
          save_interval=5000,
          write_interval=100,
          load_path=None):

    model.to(device)
    # reg_loss = Regularization(model=model, weight_decay=2e-4, p=2).to(device)
    total_loss = 0
    total_acc = 0
    lr_adjust_step = 0
    model_save_cnt = 0
    writer = SummaryWriter(log_dir='./HUGO_01_Model/summary')

    # train_iterator = iter(train_loader)
    # valid_iterator = iter(valid_loader)

    # 加载之前的模型
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    train_step = 0
    for epoch in range(1, EPOCHS + 1):

        for index, (images, labels) in enumerate(train_loader):
            train_step += 1

            images = images.view(-1, 256, 256, 1).to(device)
            labels = labels.view(-1, 1).to(device)
            labels = torch.squeeze(labels)
            # print('images.shape: ', images.shape, 'labels.shape: ', labels.shape)

            optimizer.zero_grad()
            model_output = model(images)
            loss = criterion(model_output, labels)
            # loss = output_loss + reg_loss(model)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # 计算accuracy
            model_label = torch.max(model_output, dim=1)[1]
            temp_acc = torch.eq(model_label, labels).sum().item()
            total_acc += temp_acc

            if train_step % valid_interval == 0:
                # 在validation set上测试一次
                validation(model, valid_loader, device, criterion, writer, train_step)

            if train_step % write_interval == 0:
                # 在writer中保存accuracy和loss的值
                temp_acc = total_acc / (write_interval * images.shape[0])
                temp_loss = total_loss / (write_interval * images.shape[0])
                print('EPOCH: %d/%d || currIter: %d || train_loss: %.6f || train_acc: %.6f' %
                      (epoch, EPOCHS, train_step, temp_loss, temp_acc))

                writer.add_scalar('train_accuracy', temp_acc, train_step)
                writer.add_scalar('train_loss', temp_loss, train_step)
                total_loss = 0
                total_acc = 0

            if train_step % save_interval == 0:
                # 保存模型
                torch.save(model.state_dict(), './HUGO_01_Model/Model_' + str(train_step) + '.pth')
                model_save_cnt += 1
                print('save cnt: %d || iterations: %d || saved model %d.pth' %
                      (model_save_cnt, train_step, model_save_cnt))

            if epoch == 80 or train_step == 120:
                lr_adjust_step += 1
                adjust_learning_rate(optimizer, 0.1, lr_adjust_step)

    writer.close()


def test(model, test_loader, device, weight_path=None):
    model.to(device)

    # 加载之前训练好的模型
    assert weight_path is not None, 'weight_path is None, please change weight_path'
    model.load_state_dict(torch.load(weight_path))

    # 二分类对应的四种结果
    TTCounter = 0
    TFCounter = 0
    FTCounter = 0
    FFCounter = 0

    # 含密图像和原始图像的数量
    TCounter = 0
    FCounter = 0

    step_cnt = 0
    model.eval()
    with torch.no_grad():
        for index, (images, labels) in enumerate(test_loader):
            step_cnt += 1
            cover_img, stego_img = images.view(-1, 256, 256, 1).to(device)
            cover_img = cover_img.unsqueeze(0)
            stego_img = stego_img.unsqueeze(0)
            # print(cover_img.shape, stego_img.shape)

            cover_label, stego_label = labels.view(-1, 1).to(device)
            cover_label = cover_label.item()
            stego_label = stego_label.item()
            # print(cover_label, stego_label)

            flag = random.randint(0, 1)

            if flag == 0:
                # 选择原始图像
                FCounter += 1
                model_output = model(cover_img)
                model_label = torch.max(model_output, dim=1)[1].item()
                if model_label == 0:
                    FFCounter += 1
                else:
                    FTCounter += 1
            else:
                # 选择含密图像
                TCounter += 1
                model_output = model(stego_img)
                model_label = torch.max(model_output, dim=1)[1].item()
                if model_label == 0:
                    TFCounter += 1
                else:
                    TTCounter += 1

            if step_cnt % 50 == 0:
                print('cnt: %d || TT: %d/%d, FF: %d/%d, TF: %d/%d, FT: %d/%d || PosCount: %d, NegCount: %d, correct: %.4f' %
                      (step_cnt,
                       TTCounter, TCounter,
                       FFCounter, FCounter,
                       TFCounter, TCounter,
                       FTCounter, FCounter,
                       TCounter, FCounter,
                       (TTCounter + FFCounter) * 1.0 / step_cnt))
        
        print('\nTOTAL RESULT: ')
        print('TT: %d/%d, FF: %d/%d, TF: %d/%d, FT: %d/%d || PosCount: %d, NegCount: %d, correct: %.4f' %
              (TTCounter, TCounter,
               FFCounter, FCounter,
               TFCounter, TCounter,
               FTCounter, FCounter,
               TCounter, FCounter,
               (TTCounter + FFCounter) * 1.0 / step_cnt))



