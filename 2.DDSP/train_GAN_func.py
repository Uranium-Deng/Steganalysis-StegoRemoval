import torch
import numpy as np
from tensorboardX import SummaryWriter


def adjust_learning_rate(optimizer, gamma, global_step):
    lr = 1e-3 * (gamma ** global_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validation(generator, discriminator, device, valid_loader, writer, train_step, criterion):
    generator.to(device)
    discriminator.to(device)
    criterion.to(device)

    total_loss_D = 0
    total_loss_G = 0
    iter_cnt = 0

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        for images, labels in valid_loader:
            iter_cnt += 1
            # images = images.to(device)  # 在encoder的min_max_scale 放入了cuda中
            valid = torch.tensor(np.ones(shape=(images.shape[0], 1)), requires_grad=False).to(device)
            fake = torch.tensor(np.zeros(shape=(images.shape[0], 1)), requires_grad=False).to(device)

            #########################
            # 计算 Discriminator 的损失
            #########################
            G_output = generator(images)
            true_prob = discriminator(labels)
            fake_prob = discriminator(G_output.detach().cpu())

            loss_real = criterion(true_prob, valid)
            loss_fake = criterion(fake_prob, fake)

            D_loss = (loss_fake + loss_real) / 2
            total_loss_D += D_loss.item()

            #########################
            # 计算 generator 的损失
            #########################

            G_loss = criterion(fake_prob, valid)
            total_loss_G += G_loss.item()

        avg_loss_D = total_loss_D / iter_cnt
        avg_loss_G = total_loss_G / iter_cnt
        writer.add_scalar('valid_generator_loss', avg_loss_G, train_step)
        writer.add_scalar('valid_discriminator_loss', avg_loss_D, train_step)
        print('currIter: %d || valid_generator_loss: %.6f || valid_discriminator_loss: %.6f' % (train_step, avg_loss_G, avg_loss_D))


def train_GAN(generator,
              discriminator,
              criterion,
              G_optimizer,
              D_optimizer,
              train_loader,
              valid_loader,
              device,
              train_interval,
              valid_interval,
              save_interval,
              load_path=None,
              n_epochs=5):

    assert load_path is not None, 'you did not load weight path'
    print('load_path: ', load_path)
    generator.load_state_dict(torch.load(load_path))
    print('load weight file successfully')

    generator.to(device)
    discriminator.to(device)
    criterion.to(device)

    total_loss_D = 0.0
    total_loss_G = 0.0
    train_step = 0
    model_save_cnt = 0
    lr_index = 0

    generator.train()
    discriminator.train()

    writer = SummaryWriter(log_dir='./model_save/GAN_Model/summary')

    for epoch in range(n_epochs):

        for images, labels in train_loader:
            train_step += 1
            print('train_step: ', train_step)

            valid = torch.tensor(np.ones(shape=(images.shape[0], 1)), requires_grad=False, dtype=torch.float32).to(device)
            fake = torch.tensor(np.zeros(shape=(images.shape[0], 1)), requires_grad=False, dtype=torch.float32).to(device)

            ##################
            # 训练 discriminator
            ##################
            generator.eval()
            discriminator.train()

            D_optimizer.zero_grad()
            G_output = generator(images)
            # print('G_output shape: ', G_output.shape)
            # print('G_output: ', G_output)
            # print('labels shape: ', labels.shape)
            # print('labels: ', labels)

            true_prob = discriminator(labels)  # 输入为cover images 原始不含隐秘信息的图片
            fake_prob = discriminator(G_output.detach().cpu())  # 输入为generator输出的假图片

            # discriminator 的损失函数 此处为原始GAN网络的损失函数，出现NAN和inf现象 故舍弃
            # D_loss = -(torch.mean(torch.log(true_prob) + torch.log(1 - fake_prob)))

            loss_real = criterion(true_prob, valid)
            loss_fake = criterion(fake_prob, fake)
            D_loss = (loss_fake + loss_real) / 2
            print('D_loss: ', D_loss)

            total_loss_D += D_loss.item()

            D_loss.backward()
            D_optimizer.step()

            ##################
            # 训练 generator
            ##################
            generator.train()
            discriminator.eval()

            # 原始的GAN 损失函数 出现NAN 故舍弃
            # G_output = generator(images)
            # fake_prob = discriminator(G_output.detach().cpu())  # 输入为generator输出的假图片
            # G_loss = torch.mean(torch.log(1 - fake_prob))

            G_optimizer.zero_grad()

            G_output = generator(images)
            fake_prob = discriminator(G_output.detach().cpu())
            GAN_loss = criterion(fake_prob, valid)
            content_loss = criterion(G_output, labels)
            G_loss = content_loss + 0.001 * GAN_loss
            print('G_loss: ', G_loss)
            total_loss_G += G_loss.item()

            G_loss.backward()
            G_optimizer.step()

            if train_step % valid_interval == 0:
                validation(generator, discriminator, device, valid_loader, writer, train_step, criterion)

            if train_step % train_interval == 0:
                temp_loss_G = total_loss_G / train_interval
                temp_loss_D = total_loss_D / train_interval
                print('EPOCH: %d/%d || currIter: %d, generator_loss: %.6f, discriminator_loss: %.6f' %
                      (epoch, n_epochs, train_step, temp_loss_G, temp_loss_D))
                writer.add_scalar('train_generator_loss', temp_loss_G, train_step)
                writer.add_scalar('train_discriminator_loss', temp_loss_D, train_step)
                total_loss_G = 0
                total_loss_D = 0

            if train_step % save_interval == 0:
                torch.save(generator.state_dict(), './model_save/GAN_Model/G_Model_' + str(train_step) + '.pth')
                torch.save(discriminator.state_dict(), './model_save/GAN_Model/D_Model_' + str(train_step) + '.pth')
                model_save_cnt += 1
                print('save cnt: %d || currIter: %d || save GAN_Model %d.pth' % (model_save_cnt, train_step, model_save_cnt))

        if epoch == 4:
            lr_index += 1
            adjust_learning_rate(D_optimizer, 0.1, lr_index)
            adjust_learning_rate(G_optimizer, 0.1, lr_index)



