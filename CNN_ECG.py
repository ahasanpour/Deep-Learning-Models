import numpy as np
import tensorflow as tf
from scipy.io import loadmat,savemat
#tensorflow cnn for classification ECG signals 
# DataLoader class
class DataLoader(object):
    def __init__(self):
        matfile = loadmat('/home/ahp/PycharmProjects/tensorflow/ECG200TrainTest.mat', squeeze_me=True,
                          struct_as_record=False)
        data1 = matfile['ECG200TRAIN']
        self.TrainData=data1[:,1:97]
        self.TrainTarget=data1[:,0]
        data1=[]
        data1 = matfile['ECG200TEST']
        self.TestData = data1[:, 1:97]
        self.TestTarget = data1[:, 0]

        self.num = self.TrainData.shape[0]
        self.h = 1
        self.w = 96
        self.c = 1

        self._idx = 0
    #sdfsdf
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.h, self.w, self.c))
        labels_batch = np.zeros(batch_size)
        for i in range(batch_size):
            images_batch[i, ...] = self.TrainData[self._idx].reshape((self.h, self.w, self.c))
            labels_batch[i, ...] = self.TrainTarget[self._idx]

            self._idx += 1
            if self._idx == self.num:
                self._idx = 0

        return images_batch, labels_batch

    def load_test(self):
        return self.TestData.reshape((-1, self.h, self.w, self.c)), self.TestTarget

def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_bias(shape):
        return tf.Variable(tf.zeros(shape))

def cnn(x, keep_dropout):
    weights = {
        'wc1': init_weights([1, 5, 1, 50]),  # 1x5x1 conv, 50 outputs
        'wc2': init_weights([1, 5, 50, 40]),  # 1x5x50 conv, 40 outputs
        'wc3': init_weights([1, 3, 40, 20]),  # 1x3x40 conv, 20 outputs
        'wo': init_weights([480, 2]),  # FC 1*24*20 inputs, 1024 outputs

    }
    biases = {
        'bc1': init_bias(50),
        'bc2': init_bias(40),
        'bc3': init_bias(20),
        'bo': init_bias(2),
    }

        # Conv + ReLU + Pool
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 1, 2, 1], strides=[1, 1,2, 1], padding='SAME')

        # Conv + ReLU + Pool
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']))
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

        # Conv + ReLU + Pool
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(tf.nn.bias_add(conv3, biases['bc3']))
    fc3 = tf.reshape(conv3, [-1, weights['wo'].get_shape().as_list()[0]])


        # Output FC
    out = tf.add(tf.matmul(fc3, weights['wo']), biases['bo'])

    return out

def test_Model():
        # Parameters
    learning_rate = 0.001
    training_iters = 1000
    batch_size = 10
    step_display = 10
    step_save = 500
    path_save = 'convnet'

        # Network Parameters
    h = 1  #  data input (img shape: 1*97)
    w = 96
    c = 1
    dropout = 1  # Dropout, probability to keep units
    # Construct dataloader
    loader = DataLoader()

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, h, w, c])
    y = tf.placeholder(tf.int64, None)
    keep_dropout = tf.placeholder(tf.float32)

    # Construct model
    logits = cnn(x, keep_dropout)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(logits, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # define initialization
    init = tf.global_variables_initializer()

    # define saver
    saver = tf.train.Saver()

    # define summary writer
    #writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

    # Launch the graph
    with tf.Session() as sess:
        # Initialization
        sess.run(init)

        step = 1
        while step < training_iters:
            # Load a batch of data
            images_batch, labels_batch = loader.next_batch(batch_size)

            # Run optimization op (backprop)
            sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout})

            if step % step_display == 0:
                # Calculate batch loss and accuracy while training
                l, acc = sess.run([loss, accuracy], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.})
                print ("Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(l) + ", Training Accuracy= " + \
                      "{:.4f}".format(acc))

            step += 1

            # Save model
            if step % step_save == 0:
                saver.save(sess, path_save, global_step=step)
                print ("Model saved at Iter %d !" % (step))

        print ("Optimization Finished!")

        # Calculate accuracy for 500 EKG samples
        images_test, labels_test = loader.load_test()
        print( "Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: images_test[:100],
                                          y: labels_test[:100],
                                          keep_dropout: 1.}))
if __name__ == '__main__':
    test_Model()
