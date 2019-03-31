
import numpy as np
import tensorflow as tf
import cv2
import os

# read data

row = 224
col = 224


def prep_data(datadir):
    length = len(datadir)
    x_train = np.ndarray((length, row, col, 3), dtype=np.float32)
    num = 0
    for i in datadir:
        im = cv2.imread(i)
        im = cv2.resize(im, (row, col), interpolation=cv2.INTER_CUBIC)
        x_train[num] = im/255.0
        num += 1
        if num % 200 == 0:
            print('processed {} of {}'.format(num, length))
    return x_train

# pre-process data

print('--------- start pre-process data----------')
train_Dir = 'C:/Users/tianping/Desktop/train/'
train_dirname_dog = [train_Dir + i for i in os.listdir(r'C:\Users\tianping\Desktop\train') if 'dog' in i]
train_dirname_cat = [train_Dir + i for i in os.listdir(r'C:\Users\tianping\Desktop\train') if 'cat' in i]
num_train = 1000
train_dirname = train_dirname_dog[0:int(num_train/2)]
train_dirname.extend(train_dirname_cat[0:int(num_train/2)])
x_train = prep_data(train_dirname)
y_train = np.ndarray([num_train], dtype=np.float32)
y_train[0:int(num_train/2)] = 1.0
y_train[int(num_train/2):num_train] = 0.0
shuffle = np.random.permutation(num_train)
x_train = x_train[shuffle]
y_train = y_train[shuffle]


x_val = x_train[(num_train-10):num_train]
y_val = y_train[(num_train-10):num_train]
print('-------------pre-process data is done-------------')

# param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
print('--------------start build model-------------')
with tf.name_scope('input'):
    rgb = tf.placeholder(shape=[None, row, col, 3], dtype=tf.float32)
    y = tf.placeholder(dtype=tf.float32)
    rgb_scaled = rgb * 255.0
    # Convert RGB to BGR
    VGG_MEAN = [103.939, 116.779, 123.68]
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(axis=3, values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
data_dict = np.load(file=r'G:\python file\transfer_learn\vgg16.npy', encoding='latin1').item()
print("npy file loaded")


def get_conv_filter(name):
    return tf.constant(data_dict[name][0], name="filter")


def get_bias(name):
    return tf.constant(data_dict[name][1], name="biases")


def get_fc_weight(name):
    return tf.constant(data_dict[name][0], name="weights")


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer(bottom, name):
    with tf.name_scope(name):
        filt = get_conv_filter(name)

        conv = tf.nn.conv2d(bottom, filt, strides=[1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu

def fc_layer(bottom, name):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = get_fc_weight(name)
        biases = get_bias(name)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc

conv1_1 = conv_layer(bgr, "conv1_1")
conv1_2 = conv_layer(conv1_1, "conv1_2")
pool1 = max_pool(conv1_2, 'pool1')

conv2_1 = conv_layer(pool1, "conv2_1")
conv2_2 = conv_layer(conv2_1, "conv2_2")
pool2 = max_pool(conv2_2, 'pool2')

conv3_1 = conv_layer(pool2, "conv3_1")
conv3_2 = conv_layer(conv3_1, "conv3_2")
conv3_3 = conv_layer(conv3_2, "conv3_3")
pool3 = max_pool(conv3_3, 'pool3')

conv4_1 = conv_layer(pool3, "conv4_1")
conv4_2 = conv_layer(conv4_1, "conv4_2")
conv4_3 = conv_layer(conv4_2, "conv4_3")
pool4 = max_pool(conv4_3, 'pool4')

conv5_1 = conv_layer(pool4, "conv5_1")
conv5_2 = conv_layer(conv5_1, "conv5_2")
conv5_3 = conv_layer(conv5_2, "conv5_3")
pool5 = max_pool(conv5_3, 'pool5')

fc6 = fc_layer(pool5, "fc6")
assert fc6.get_shape().as_list()[1:] == [4096]
relu6 = tf.nn.relu(fc6)

with tf.name_scope('fc7'):
    fc7_w = tf.Variable(tf.truncated_normal([4096, 256], stddev=0.1, dtype=tf.float32))
    fc7_b = tf.Variable(tf.zeros([256], dtype=tf.float32))
    fc7 = tf.nn.relu(tf.matmul(relu6, fc7_w) + fc7_b)
    tf.summary.histogram('fc7_w', fc7_w)
    tf.summary.histogram('fc7_b', fc7_b)

with tf.name_scope('out'):
    out_w = tf.Variable(tf.truncated_normal([256, 1], dtype=tf.float32))
    out_b = tf.Variable(tf.zeros([1], dtype=tf.float32))
    outputs = tf.matmul(fc7, out_w) + out_b
    tf.summary.histogram('out_w', out_w)
    tf.summary.histogram('out_b', out_b)

with tf.name_scope('train'):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs,
                                                                   labels=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    tf.summary.scalar('loss', loss)


sess = tf.Session()
writer_train = tf.summary.FileWriter('G:/python file/logs/train/', sess.graph)
init_op = tf.global_variables_initializer()
sess.run(init_op)
print('------------end build model------------')

print('----------------training---------------')

for i in range(50):
    sess.run(train_step, feed_dict={rgb: x_train[(6*i):(6*(i+1))],
                                    y: y_train[(6*i):(6*(i+1))]})
    print(sess.run(loss, feed_dict={rgb: x_train[(6*i):(6*(i+1))],
                                    y: y_train[(6*i):(6*(i+1))]}))
    print('processed %s of 100' % (i+1))
