import tensorflow as tf
import threading
import numpy as np

class Data(object):
    def __init__(self, session, coord, batch_size=32):
        self.session = session
        self.coord = coord
        self.batch_size = batch_size
        self.lock = threading.Lock() # to avoid exception when different threads call the same iterator at the same time

        self.image = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        self.my_iter = self._create_iterator() # create the data iterator
        self._create_queue() # create the queue

    def _create_queue(self):
        queue = tf.RandomShuffleQueue(capacity=10*self.batch_size,
                              min_after_dequeue=4*self.batch_size,
                              dtypes=[tf.float32, tf.float32],
                             shapes=[[28, 28, 1], [10]])
        # The FIFOQueue  below is simply to demonstrate how the iterator
        # is shared accord multiple threads
        #queue = tf.FIFOQueue(capacity=10*batch_size,
        #                      dtypes=[tf.float32, tf.float32],
        #                     shapes=[[28, 28, 1], [10]])
        self.enqueue_op = queue.enqueue_many([self.image, self.label])
        self.image_batch, self.label_batch = queue.dequeue_many(self.batch_size)

    def _create_iterator(self):
        for i in range(100):
            lab = np.ones((self.batch_size, 10)) * i
            im = np.ones((self.batch_size, 28, 28, 1)) * i
            yield (im, lab)

    def _enqueue_data(self):
        with self.lock:
            while True:
                im, lab = next(self.my_iter) # iterator shared over all threads
                im, lab = data_augmentation(im, lab) 
                self.session.run(self.enqueue_op, feed_dict={self.image: im, self.label:lab})

    def create_threads(self, num_threads=1):
        self.threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=self._enqueue_data, args=())
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)


def data_augmentation(image, label):
    """
    Perform all data augmentation with numpy
    operations here.
    """
    return image, label


batch_size = 8
with tf.Session() as session:
    coord = tf.train.Coordinator()
    data = Data(session, coord, batch_size)
    data.create_threads(5) # create 5 threads

    # train network
    for i in range(10):
        lab = session.run([data.label_batch])
        print(lab)
