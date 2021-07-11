import torch
from tensorboardX import SummaryWriter


def adjust_learning_rate(optimizer, gamma, global_step):
    lr = 1e-3 * (gamma ** global_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validation(model, device, valid_loader, writer, train_step, criterion):
    model.to(device)
    criterion.to(device)

    total_loss = 0
    img_cnt = 0

    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            img_cnt += images.shape[0]
            # images = images.to(device)  # 在encoder的min_max_scale 放入了cuda中
            labels = labels.to(device)

            model_output = model(images).to(device)
            loss = criterion(model_output, labels)
            total_loss += loss.item()

        avg_loss = total_loss / img_cnt
        writer.add_scalar('valid_loss', avg_loss, train_step)
        print('currIter: %d || valid_loss: %.6f' % (train_step, avg_loss))


def train(model,
          train_loader,
          valid_loader,
          optimizer,
          criterion,
          device,
          n_epochs,
          train_interval,
          valid_interval,
          save_interval,
          load_path=None):

    print('in train function')
    model.to(device)
    criterion.to(device)

    if load_path is not None:
        print('load_path: ', load_path)
        model.load_state_dict(torch.load(load_path))
        print('load weight file successfully')

    writer = SummaryWriter(log_dir='./model_save/HUGO_1_Model/summary')
    model_save_cnt = 0
    total_loss = 0.0
    train_step = 0
    lr_index = 0

    print('start training')
    for epoch in range(1, n_epochs + 1):

        for images, labels in train_loader:
            train_step += 1
            print('train_step: ', train_step)

            # images = images.to(device)  # 输入的原始图片不要放入cuda中，min_max_scale 标准化之后已经放入cuda中
            labels = labels.to(device)

            optimizer.zero_grad()
            model_output = model(images)

            # print('model_output: ', model_output)
            # print('model_output.shape: ', model_output.shape)
            # print('labels: ', labels)
            # print('labels shape: ', labels.shape)

            # 论文中的值是5点多
            loss = criterion(model_output, labels)  # 这里的loss就是针对这个batch中图片每一个像素点的平均像素点的值的平方

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if train_step % valid_interval == 0:
                validation(model, device, valid_loader, writer, train_step, criterion)

            if train_step % train_interval == 0:
                temp_loss = total_loss / train_interval
                print('EPOCH: %d/%d || currIter: %d, train_loss: %.6f' % (epoch, n_epochs, train_step, temp_loss))
                writer.add_scalar('train_loss', temp_loss, train_step)
                total_loss = 0

            if train_step % save_interval == 0:
                torch.save(model.state_dict(), './model_save/HUGO_1_Model/Model_' + str(train_step) + '.pth')
                model_save_cnt += 1
                print('save cnt: %d || currIter: %d || save model %d.pth' % (model_save_cnt, train_step, model_save_cnt))

        if epoch == 8 or epoch == 12:
            lr_index += 1
            adjust_learning_rate(optimizer, 0.1, lr_index)

    writer.close()
