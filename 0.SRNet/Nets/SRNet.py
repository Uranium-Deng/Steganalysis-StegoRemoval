import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from Nets.Models import Model


class SRNet(Model):
    # 继承自Model, 重写_build_model函数
    def build_model(self, inputs):
        self.input = inputs

        # 数据格式的转换，将[batch_size, height, width, channels] -> [batch_size, channels, height, width]
        if self.data_format == 'NCHW':
            # 网络规定输入的数据类型是NCHW,但若输入的格式是NHWC,需要进行通道转换
            reduction_axis = [2, 3]
            _inputs = tf.cast(tf.transpose(inputs, [0, 3, 1, 2]), tf.float32)
        else:
            # 网络规定输入的数据格式不是NCHW而是NHWC,与输入格式相同,故不要进行通道转换
            reduction_axis = [1, 2]
            _inputs = tf.cast(inputs, tf.float32)

        with arg_scope([layers.conv2d],
                       num_outputs=16,  # 卷积核的数量
                       kernel_size=3,  # 卷积核的大小 [kernel_size, kernel_size]
                       stride=1,  # 移动步长
                       padding='SAME',  # padding类型选择SAME
                       data_format=self.data_format,  # 数据类型： [batch_size, channels, height, width]
                       activation_fn=None,  # 激活函数
                       weights_initializer=layers.variance_scaling_initializer(),  # 卷积核权重初始化
                       weights_regularizer=layers.l2_regularizer(2e-4),  # 卷积核权重正则
                       biases_initializer=tf.constant_initializer(0.2),  # 卷积核bias初始化
                       biases_regularizer=None), \
             arg_scope([layers.batch_norm],
                       decay=0.9,  # 衰减率
                       center=True,  # beta 偏移量
                       scale=True,  # 乘以 gamma
                       updates_collections=None,
                       is_training=self.is_training,
                       fused=True,
                       data_format=self.data_format), \
             arg_scope([layers.avg_pool2d],
                       kernel_size=3,
                       stride=2,
                       padding='SAME',
                       data_format=self.data_format):
            # 开始之前shape=[n, 1, 256, 256]

            # 第一种结构类型
            with tf.variable_scope('Layer1'):
                conv = layers.conv2d(_inputs, num_outputs=64)
                actv = tf.nn.relu(layers.batch_norm(conv))  # [n, 64, 256, 256]
            with tf.variable_scope('Layer2'):
                conv = layers.conv2d(actv)
                actv = tf.nn.relu(layers.batch_norm(conv))  # [n, 16, 256, 256]

            # 第二种结构类型
            with tf.variable_scope('Layer3'):
                conv1 = layers.conv2d(actv)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn2 = layers.batch_norm(conv2)
                res = tf.add(actv, bn2)  # [n, 16, 256, 256]
            with tf.variable_scope('Layer4'):
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn2 = layers.batch_norm(conv2)
                res = tf.add(res, bn2)  # [n, 16, 256, 256]
            with tf.variable_scope('Layer5'):
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn2 = layers.batch_norm(conv2)
                res = tf.add(res, bn2)  # [n, 16, 256, 256]
            with tf.variable_scope('Layer6'):
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn2 = layers.batch_norm(conv2)
                res = tf.add(res, bn2)  # [n, 16, 256, 256]
            with tf.variable_scope('Layer7'):
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn2 = layers.batch_norm(conv2)
                res = tf.add(res, bn2)  # [n, 16, 256, 256]

            # 第三种结构类型
            with tf.variable_scope('Layer8'):
                convs = layers.conv2d(res, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)  # [n, 16, 128, 128]
            with tf.variable_scope('Layer9'):
                convs = layers.conv2d(res, num_outputs=64, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res, num_outputs=64)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=64)
                bn2 = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn2)
                res = tf.add(convs, pool)  # [n, 64, 64, 64]
            with tf.variable_scope('Layer10'):
                convs = layers.conv2d(res, num_outputs=128, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res, num_outputs=128)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=128)
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)  # [n, 128, 32, 32]
            with tf.variable_scope('Layer11'):
                convs = layers.conv2d(res, num_outputs=256, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res, num_outputs=256)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=256)
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)  # [n, 256, 16, 16]

            # 第四种结构类型
            with tf.variable_scope('Layer12'):
                conv1 = layers.conv2d(res, num_outputs=512)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=512)
                bn = layers.batch_norm(conv2)  # [n, 512, 16, 16]
                avgp = tf.reduce_mean(bn, reduction_axis, keepdims=True)  # [n, 512, 1, 1] -> [n, 512]

        # 全连接层[n, 512] x [512, 2] -> [n, 2]
        ip = layers.fully_connected(layers.flatten(avgp),
                                    num_outputs=2,
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                                    biases_initializer=tf.constant_initializer(0.),
                                    scope='ip')
        # [n, 2]
        self.outputs = ip
        # print('self.outputs.shape: ', self.outputs.shape)
        return self.outputs
