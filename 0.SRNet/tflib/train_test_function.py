from tflib.queue import GeneratorRunner, queueSelection
from Nets.Models import AverageSummary
import time
import tensorflow as tf
from Nets.SRNet import SRNet
from glob import glob


def train(model_class,  # SRNet模型
          train_gen,  # train的generator
          valid_gen,  # valid的generator
          train_batch_size,  # 训练集的batch_size
          valid_batch_size,  # 测试集的batch_size
          train_ds_size,  # 训练集的大小
          valid_ds_size,  # 测试集的大小
          optimizer,  # 优化器 Adamax
          boundaries,  # 学习率变化的边界
          values,  # 学习率的值
          train_interval,  # 每train_interval个iterations就在tensorboard中记录loss和accuracy的平均值
          valid_interval,  # 每隔valid_interval个iterations就在validation set上测试一遍
          max_iter,  # 最大迭代次数 500k
          save_interval,  # 保存模型的间隔iterations
          log_path,  # 模型保存的路径
          num_runner_threads,  # 训练向队列塞入数据的线程数量
          load_path=None):  # 之前训练好的模型所在的路径

    tf.reset_default_graph()
    # train和validation中的runner
    train_runner = GeneratorRunner(train_gen, train_batch_size * 10)
    valid_runner = GeneratorRunner(valid_gen, valid_batch_size * 10)
    print('train_runner & valid_runner down successfully!')

    is_training = tf.get_variable(name='is_training', dtype=tf.bool, initializer=True, trainable=False)

    if train_batch_size == valid_batch_size:
        batch_size = train_batch_size
        disable_training_op = tf.assign(is_training, False)
        enable_training_op = tf.assign(is_training, True)
    else:
        batch_size = tf.get_variable(name='batch_size', dtype=tf.int32,
                                     collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                     initializer=train_batch_size, trainable=False)

        disable_training_op = tf.group(tf.assign(batch_size, valid_batch_size),
                                       tf.assign(is_training, False))
        enable_training_op = tf.group(tf.assign(batch_size, train_batch_size),
                                      tf.assign(is_training, True))

        # 选择runner从队列中拿出batch_size个元素, (?, 256, 256, 1), (?,)
        img_batch, label_batch = queueSelection([valid_runner, train_runner],
                                                tf.cast(is_training, tf.int32), batch_size)

        # 构建网络模型
        model = model_class(is_training=is_training, data_format='NCHW')
        model.build_model(img_batch)
        print('build model successfully!')

        # 获得模型的loss和accuracy
        loss, accuracy = model.build_loss(label_batch)
        print('get loss and accuracy successfully!')

        train_loss_s = AverageSummary(loss, name='train_loss', num_iterations=train_interval)
        train_accuracy_s = AverageSummary(accuracy, name='train_accuracy', num_iterations=train_interval)

        valid_loss_s = AverageSummary(loss, name='valid_loss',
                                      num_iterations=float(valid_ds_size) / float(valid_batch_size))
        valid_accuracy_s = AverageSummary(accuracy, name='valid_accuracy',
                                          num_iterations=float(valid_ds_size) / float(valid_batch_size))

        # iteration从0开始计数, 主要是为了lr的调整而服务
        global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                      initializer=tf.constant_initializer(65000), trainable=False)

        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        lr_summary = tf.summary.scalar('learning_rate', learning_rate)
        optimizer = optimizer(learning_rate)

        # 定义各种操作
        minimize_op = optimizer.minimize(loss, global_step)
        # 训练
        train_op = tf.group(minimize_op, train_loss_s.increment_op, train_accuracy_s.increment_op)
        # 测试
        increment_valid = tf.group(valid_loss_s.increment_op, valid_accuracy_s.increment_op)
        # 初始化
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5000)

        print('session start!!!')
        train_cnt = 1
        valid_cnt = 1
        model_save_cnt = 1

        # session正式开始！！！
        with tf.Session() as sess:
            # 初始化全局变量和局部变量
            sess.run(init_op)

            # 若之前训练好的ckpt文件存在，则加载该文件
            if load_path is not None:
                print('load_path: ', load_path)
                saver.restore(sess, load_path)

            # 开启线程，向队列中放入数据
            train_runner.start_threads(sess, num_threads=num_runner_threads)
            valid_runner.start_threads(sess, num_threads=1)

            # 指定Graph保存的路径
            writer = tf.summary.FileWriter(log_path + '/LogFile/', sess.graph)
            start = sess.run(global_step)

            # 先重置，然后计算训练前，测试集上的平均准确率
            sess.run(disable_training_op)
            sess.run([valid_loss_s.reset_variable_op,
                      valid_accuracy_s.reset_variable_op,
                      train_loss_s.reset_variable_op,
                      train_accuracy_s.reset_variable_op])
            _time = time.time()
            for i in range(0, valid_ds_size, valid_batch_size):
                sess.run([increment_valid])
            _acc_val = sess.run(valid_accuracy_s.mean_variable)
            print('initial accuracy in validation set: ', _acc_val)
            print('evaluation time on validation set: ', time.time() - _time, ' seconds')

            # 记录valid_loss和valid_accuracy的平均值，同时重置sum_variable的值
            valid_loss_s.add_summary(sess, writer, start)
            valid_accuracy_s.add_summary(sess, writer, start)

            # 开始训练了！！！
            sess.run(enable_training_op)
            print('network will be evaluated in validation set every %d iterations' % valid_interval)
            for i in range(start + 1, max_iter + 1):
                # 这里有必要增加一些输出，不然整个训练过程中没有输出，不知道具体进度
                sess.run(train_op)

                if i % train_interval == 0:
                    print('train cnt: %f || iterations: %f || train accuracy: %f' % (
                    train_cnt, i, sess.run(train_accuracy_s.mean_variable)))
                    train_cnt += 1
                    train_loss_s.add_summary(sess, writer, i)
                    train_accuracy_s.add_summary(sess, writer, i)
                    s = sess.run(lr_summary)
                    writer.add_summary(s, i)

                if i % valid_interval == 0:
                    sess.run(disable_training_op)

                    for j in range(0, valid_ds_size, valid_batch_size):
                        sess.run([increment_valid])
                    print('validation cnt: %d || iterations: %d || validation accuracy: %d' % (
                    valid_cnt, i, sess.run(valid_accuracy_s.mean_variable)))
                    valid_cnt += 1
                    valid_loss_s.add_summary(sess, writer, i)
                    valid_accuracy_s.add_summary(sess, writer, i)

                    sess.run(enable_training_op)

                if i % save_interval == 0:
                    print('save cnt: %d || iterations: %d || saved model %d.ckpt' % (model_save_cnt, i, i))
                    model_save_cnt += 1
                    saver.save(sess, log_path + '/Model_' + str(i) + '.ckpt')


def test(model_class,  # 模型
         gen,  # generator
         load_path,  # 训练好的模型所在的路径
         batch_size,  # 测试时采用的batch_size
         ds_size):  # 数据集的大小

    # 在测试集上测试，输出accuracy和loss

    tf.reset_default_graph()

    runner = GeneratorRunner(gen, batch_size * 10)
    img_batch, label_batch = runner.get_batch_inputs(batch_size)

    model = model_class(is_training=False, data_format='NCHW')
    model.build_model(img_batch)
    loss, accuracy = model.build_loss(label_batch)

    loss_summary = AverageSummary(loss, name='loss', num_iterations=float(ds_size) / float(batch_size))
    accuracy_summary = AverageSummary(accuracy, name='accuracy', num_iterations=float(ds_size) / float(batch_size))

    increment_op = tf.group(loss_summary.increment_op, accuracy_summary.increment_op)
    global_step = tf.get_variable(name='global_step', shape=[], dtype=tf.int32,
                                  initializer=tf.constant_initializer(0), trainable=False)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=10000)

    with tf.Session() as sess:
        sess.run(init_op)  # 变量初始化
        saver.restore(model, load_path)  # 加载训练好的模型
        runner.start_threads(sess, num_threads=1)  # 开启线程

        for i in range(0, ds_size, batch_size):
            sess.run([increment_op])
        mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable, accuracy_summary.mean_variable])
        print('Accuracy: ', mean_accuracy, ' | Loss: ', mean_loss)


def get_confusion_matrix(gen,  # generator
                         weight_path,  # 训练好的模型所在的路径
                         data_size,  # 数据集的大小
                         batch_size=1):  # 测试时采用的batch_size

    tf.reset_default_graph()

    assert weight_path is not None, 'weight_path is None, please change weight_path'
    # 二分类对应的四种结果
    TTCounter = 0
    TFCounter = 0
    FTCounter = 0
    FFCounter = 0

    # 含密图像和原始图像的数量
    TCounter = 0
    FCounter = 0

    step_cnt = 0
    model = SRNet(is_training=False, data_format='NCHW')
    print('SRNet model successfully')

    runner = GeneratorRunner(gen, batch_size * 10)
    img_batch, label_batch = runner.get_batch_inputs(batch_size)
    model_output = model.build_model(img_batch)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)

    with tf.Session() as sess:
        sess.run(init_op)
        model_file = tf.train.latest_checkpoint(weight_path)
        saver.restore(sess, model_file)
        runner.start_threads(sess, num_threads=1)

        for step in range(0, data_size, batch_size):
            # 拿出来的图像顺序就是cover, stego, cover, stego 这样的顺序
            step_cnt += 1
            model_label = sess.run(tf.argmax(model_output, 1))[0]

            if step_cnt % 2 == 1:
                # 原始图像
                FCounter += 1
                if model_label == 0:
                    FFCounter += 1
                else:
                    FTCounter += 1
            else:
                # 含密图像
                TCounter += 1
                if model_label == 0:
                    TFCounter += 1
                else:
                    TTCounter += 1

            if step_cnt % 50 == 0:
                print('cnt: %d || TT: %d/%d, FF: %d/%d, TF: %d/%d, FT: %d/%d || PosCount: %d, NegCount: %d, correct: '
                      '%.4f' % (step_cnt,
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

