import tensorflow as tf


class AverageSummary:
    def __init__(self, variable, name, num_iterations):
        """
        :param variable: tensor变量, loss或者accuracy
        :param name: train_loss, train_accuracy, valid_loss, valid_accuracy 四选一
        :param num_iterations: 迭代的次数
        """
        # 对训练集而言, 每num_iterations(train_interval)个iteration就求一次平均值，然后在图中记录
        self.sum_variable = tf.get_variable(name=name,
                                            shape=[],
                                            initializer=tf.constant_initializer(0),
                                            dtype=variable.dtype.base_dtype,
                                            trainable=False,
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])
        with tf.control_dependencies([variable]):
            self.increment_op = tf.assign_add(self.sum_variable, variable)
        # 平均值
        self.mean_variable = self.sum_variable / float(num_iterations)
        # 生成平均值的标量图, 类似的共有4个: train_loss, train_accuracy, valid_loss, valid_accuracy
        self.summary = tf.summary.scalar(name, self.mean_variable)
        # 将summary重置为0
        with tf.control_dependencies([self.summary]):
            self.reset_variable_op = tf.assign(self.sum_variable, 0)

    def add_summary(self, sess, writer, step):
        s, _ = sess.run([self.summary, self.reset_variable_op])
        writer.add_summary(s, step)


class Model:
    # 定义整个模型
    def __init__(self, is_training=None, data_format='NCHW'):
        """
        :param is_training: 训练标志位
        :param data_format: 规定网络输入数据的格式NCHW
        """
        self.data_format = data_format  # 规定模型中数据格式为NCHW [batch_size, channels, height, width]
        if is_training is None:
            self.is_training = tf.get_variable(name='is_training', dtype=tf.bool,
                                               initializer=tf.constant_initializer(True), trainable=False)
        else:
            self.is_training = is_training

        self.label = None  # 真正的label
        self.input = None  # 网路的输入
        self.outputs = None  # SRNet网络的输出
        self.loss = None  # 模型的损失
        self.accuracy = None  # 模型的准确率

    def build_model(self, inputs):
        # 子类在这里定义模型
        raise NotImplementedError('Here is your model definition')

    def build_loss(self, label):
        # 传入真正的labels, 结合self.outputs计算出模型的loss和accuracy
        self.label = tf.cast(label, tf.int64)
        with tf.variable_scope('loss'):
            one_hot = tf.one_hot(self.label, 2)

            # 网络输出和label之间的损失
            output_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=self.outputs))
            # 网络参数的第二正则损失，防止过拟合
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            # 总的损失
            self.loss = tf.add_n([output_loss] + reg_loss)

        with tf.variable_scope('accuracy'):
            temp = tf.argmax(self.outputs, 1)  # 找到最大的值所在的下标
            equal = tf.equal(temp, self.label)
            self.accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

        return self.loss, self.accuracy

