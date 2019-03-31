import sys
sys.path.append('G:/python_file/transfer_learn/')

import vgg16
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import time

content_path = 'G:/python_file/transfer_learn/Koala.jpg'
style_path = 'G:/python_file/transfer_learn/sky.jpg'


def prepocess(image_path):

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)

    image_array = np.ndarray([1, 224, 224, 3], dtype=np.float32)
    image_array[0] = image / 255.0

    return image_array



# ------------------------calculate loss------------------- #


def mean_square_error(a, b):
    return tf.reduce_mean(tf.square(a - b))


def content_loss(content_layers, content_values_tf):
    layer_losses = []
    for x_value, value in zip(content_layers, content_values_tf):
        loss = mean_square_error(x_value, value)
        layer_losses.append(loss)
    return tf.reduce_mean(layer_losses)

def gram_mat(tensor):
    n_c = int(tensor.get_shape()[3])
    n_h = int(tensor.get_shape()[1])
    n_w = int(tensor.get_shape()[2])
    matrix = tf.reshape(tensor, [-1, n_c])
    return tf.matmul(tf.transpose(matrix), matrix)/n_h/n_w


def style_loss(style_layers, style_values_gram_tf):
    layer_losses = []
    for x_value, value in zip(style_layers, style_values_gram_tf):
        loss = mean_square_error(gram_mat(x_value), value)
        n_c = int(x_value.get_shape()[3])
        layer_losses.append(loss/n_c**2)            # add weights
    return tf.reduce_mean(layer_losses)

#########################################################################


def style_content_initial_values(sess, style_layers, content_layers):
    """
    calculate the initial values of style layers and content layers
    they won't change later
    """
    style_values = sess.run(style_layers,
                        feed_dict={image: style_image})
    style_values_gram_tf = [gram_mat(tf.constant(value)) for value in style_values]
    content_values = sess.run(content_layers,
                          feed_dict={image: content_image})
    content_values_tf = [tf.constant(value) for value in content_values]
    return style_values_gram_tf, content_values_tf


print('---------start build model--------------')
content_image = prepocess(content_path)
style_image = prepocess(style_path)




with tf.Session() as sess:
    image = tf.placeholder(shape=[1, 224, 224, 3], dtype=tf.float32, name='input')

    model = vgg16.Vgg16(vgg16_npy_path=r'G:\python_file\transfer_learn\vgg16.npy')
    model.build(image)
    style_layers = [model.conv1_1, model.conv2_1, model.conv3_1, model.conv4_1, model.conv5_1]
    content_layers = [model.conv5_3]

    style_values_gram_tf, content_values_tf = style_content_initial_values(sess, style_layers, content_layers)
print('-----------end build model--------------')


print('--------------start build model again by x---------------')

x = tf.Variable(content_image, dtype=tf.float32)  # initialize by content image
style_w = tf.placeholder(dtype=tf.float32)
lr = tf.placeholder(dtype=tf.float32)

model_x = vgg16.Vgg16(vgg16_npy_path=r'G:\python_file\transfer_learn\vgg16.npy')
model_x.build(x)

style_layers = [model_x.conv1_1, model_x.conv2_1, model_x.conv3_1, model_x.conv4_1, model_x.conv5_1]
content_layers = [model_x.conv5_3]
style_losses = style_loss(style_layers, style_values_gram_tf)
content_losses = content_loss(content_layers, content_values_tf)
total_loss = style_w * style_losses + 1 * content_losses
step = tf.train.AdamOptimizer(lr).minimize(total_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('--------------end build model again by x---------------')


for i in range(1000):
    if (i+1) % 10 == 0:
        my_image = sess.run(x)
        my_image = my_image[0]
        cv2.imshow('style',style_image[0])
        cv2.imshow('content', content_image[0])
        cv2.imshow('mix',np.clip(my_image, 0, 1))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    sess.run(step, feed_dict={style_w: 10, lr: 0.1})
