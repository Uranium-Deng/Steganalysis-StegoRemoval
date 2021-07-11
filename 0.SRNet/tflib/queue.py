import tensorflow as tf
import threading


class GeneratorRunner:
    """
    创建一个多线程先入先出队列，队列中包含需要输入网络的张量
    """

    def __init__(self, generator, capacity):
        """
        :param generator: 返回数据的generator
        :param capacity: 队列的最大容量
        """
        self.generator = generator
        self.capacity = capacity
        self.stop_threads = True  # 关闭线程的标志位
        self.threads = []  # 开启多线程之后，保存所有线程的列表

        _input = next(self.generator(0, 1))
        # print('type(_input): ', type(_input))
        if type(_input) is not list:
            raise ValueError('generator does not return a list, but:%s' % str(type(_input)))

        input_batch_size = _input[0].shape[0]
        if not all(input_batch_size == _input[i].shape[0] for i in range(len(_input))):
            raise ValueError('all the inputs do not have the same batch size, ' +
                             'the batch sizes are: %s' % [_input[i].shape[0] for i in range(len(_input))])

        # 先入先出队列的参数, type, shape, data
        self.dtype = []
        self.dshape = []
        self.data = []
        for i in range(len(_input)):
            self.dtype.append(_input[i].dtype)
            self.dshape.append(_input[i].shape[1:])
            self.data.append(tf.placeholder(shape=(input_batch_size,) + self.dshape[i], dtype=self.dtype[i]))

        # print(self.dtype)
        # print(self.dshape)
        # print(self.data)

        # 创建先入先出队列
        self.queue = tf.FIFOQueue(capacity=self.capacity, shapes=self.dshape, dtypes=self.dtype)
        # 多个元素入队列操作
        self.enqueue_op = self.queue.enqueue_many(self.data)
        # 关闭队列操作
        self.close_queue_op = self.queue.close(cancel_pending_enqueues=True)

    def get_batch_inputs(self, batch_size):
        # 从队列中弹出batch_size个元素
        return self.queue.dequeue_many(batch_size)

    def thread_function(self, sess, thread_idx, num_threads=1):
        # 每个子线程的target function
        # 函数作用: 调用generator获取数据并将数据不停的塞入队列
        # 这个函数永远不会结束，因为for循环中generator是一个死循环
        try:
            # print('feed data to queue')
            for data in self.generator(thread_idx, num_threads):
                # print('thread_idx: %d in loop' % thread_idx)
                sess.run(self.enqueue_op, feed_dict={i: d for i, d in zip(self.data, data)})
                # print('self.queue.size: ', sess.run(self.queue.size()))
                # print('after sess run')
            if self.stop_threads:
                print('in target function, stop threads')
                return
        except RuntimeError:
            print('thread function error!!!')
            pass
        print('target function end')

    def start_threads(self, sess, num_threads=1):
        # 创建num_threads个线程, sess就是tf session会话
        self.stop_threads = False
        for i in range(num_threads):
            t = threading.Thread(target=self.thread_function, args=(sess, i, num_threads))
            t.daemon = True
            t.start()
            self.threads.append(t)
        return self.threads

    def stop_runner(self, sess):
        # 关闭线程，同时关闭队列, sess就是tf session
        self.stop_threads = True
        sess.run(self.close_queue_op)


def queueSelection(runners, sel, batch_size):
    """
    :param runners: [valid_runner, train_runner] runner组成的列表
    :param sel: 选择的queue在列表中的下标index
    :param batch_size: 从队列中选择batch_size个元素
    :return: 从对应的队列中返回batch_size个元素
    """
    selection_queue = tf.FIFOQueue.from_list(sel, [r.queue for r in runners])
    return selection_queue.dequeue_many(batch_size)

