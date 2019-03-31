import vgg16
import numpy as np
import tensorflow as tf
import cv2
# import matplotlib.pyplot as plt
# import time
import sys
# 全局变量，根据自己的情况进行修改
MAIN_FILE = 'G:/python_file/Style_Transfer/'                    # 工作目录，其他的文件和图片都放在这个目录下
sys.path.append(MAIN_FILE)

CONTENT_PATH = 'G:/python_file/Style-Transfer/Koala.jpg'        # content image的位置
STYLE_PATH = 'G:/python_file/Style-Transfer/sky.jpg'            # style image的位置
VGG16_NPY_PATH = r'G:\python_file\Style-Transfer\vgg16.npy'     # vgg16.npy文件的位置
EPOCH = 1000                                                    # 运行次数
LR = 0.1                                                        # Learning Rate
WEIGHT = 10                                                     # WEIGHT * style_loss + content_loss


# 对image做预处理：标准化为[0,1]
def prepocess(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)

    image_array = np.ndarray([1, 224, 224, 3], dtype=np.float32)
    image_array[0] = image / 255.0

    return image_array

# ------------------------calculate loss------------------------ #


def mean_square_error(a, b):
    """MSE"""
    return tf.reduce_mean(tf.square(a - b))


def gram_mat(tensor):
    """Gram Matrix"""
    n_c = int(tensor.get_shape()[3])
    n_h = int(tensor.get_shape()[1])
    n_w = int(tensor.get_shape()[2])
    matrix = tf.reshape(tensor, [-1, n_c])
    return tf.matmul(tf.transpose(matrix), matrix)/n_h/n_w          # 计算gram matrix并对gram matrix标准化


def content_loss(content_layers, content_values_tf):
    """
    计算content loss
    :parameter
        content_layers: Data Type 是 List，你可以选择VGG16的某些层，最好是较高的神经层
        content_values_tf: content image在VGG16所选层的特征图
    """
    layer_losses = []
    for x_value, value in zip(content_layers, content_values_tf):
        loss = mean_square_error(x_value, value)
        layer_losses.append(loss)
    return tf.reduce_mean(layer_losses)


def style_loss(style_layers, style_values_gram_tf):
    """
    计算style loss
    :parameter
        style_layers: Data Type 是 List，你可以选择VGG16的某些层，最好是较低的神经层
        style_values_gram_tf: style image在VGG16所选层的特征图
    """
    layer_losses = []
    for x_value, value in zip(style_layers, style_values_gram_tf):
        loss = mean_square_error(gram_mat(x_value), value)
        n_c = int(x_value.get_shape()[3])
        layer_losses.append(loss/n_c**2)            # add weights
    return tf.reduce_mean(layer_losses)

#########################################################################


def style_content_initial_values(sess, style_layers, content_layers):
    """
    Calculate the initial values of style layers and content layers.They won't change later.
    :parameter
        sess: tensorflow中的会话(即tf.Session())
        style_layers: VGG16中的某些层，较低层
        content_layers: VGG16中的某些层，较高层
    :returns
        style_values_gram_tf: 得到style image在VGG16所选层的特征图(feature maps)
        content_values_tf: 得到content image在VGG16所选层的特征图(feature maps)
    """
    style_values = sess.run(style_layers,
                            feed_dict={image: style_image})
    style_values_gram_tf = [gram_mat(tf.constant(value)) for value in style_values]
    content_values = sess.run(content_layers,
                              feed_dict={image: content_image})
    content_values_tf = [tf.constant(value) for value in content_values]
    return style_values_gram_tf, content_values_tf


print('------------start build model--------------')
content_image = prepocess(CONTENT_PATH)     # content image
style_image = prepocess(STYLE_PATH)         # style image

# 计算 content image 和 style image 在所选神经层的特征图
with tf.Session() as sess:
    image = tf.placeholder(shape=[1, 224, 224, 3], dtype=tf.float32, name='input')

    model = vgg16.Vgg16(vgg16_npy_path=VGG16_NPY_PATH)
    model.build(image)
    style_layers = [model.conv1_1, model.conv2_1, model.conv3_1, model.conv4_1, model.conv5_1]      # can change
    content_layers = [model.conv5_3]                                                                # can change

    style_values_gram_tf, content_values_tf = style_content_initial_values(sess, style_layers, content_layers)
print('-----------end build model--------------')

# 下面就开始对 noise image （x）建立计算图 并 进行更新
print('--------------start build model again by x---------------')

x = tf.Variable(content_image, dtype=tf.float32)  # initialize by content image
style_w = tf.placeholder(dtype=tf.float32)        # 也就是我们的全局变量WEIGHT
lr = tf.placeholder(dtype=tf.float32)             # 全局变量LR

model_x = vgg16.Vgg16(vgg16_npy_path=VGG16_NPY_PATH)
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

for i in range(EPOCH):
    if (i+1) % 10 == 0:     # 每 10 个epoch可视化一次,出现图片之后需要按下enter，程序才能继续进行
        my_image = sess.run(x)
        my_image = my_image[0]
        cv2.imshow('style', style_image[0])
        cv2.imshow('content', content_image[0])
        cv2.imshow('mix', np.clip(my_image, 0, 1))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    _, s_l, c_l, t_l = sess.run([step, style_losses, content_losses, total_loss],
                                feed_dict={style_w: WEIGHT, lr: LR})    # update the noise image x.
    print('epoch %d> style loss is %.2f | content loss is %.2f | total loss is %.2f' %
          (i+1, s_l, c_l, t_l))
