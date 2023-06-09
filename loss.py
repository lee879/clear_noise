import tensorflow as tf
from keras.applications import VGG16
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.layers import Conv2D,LeakyReLU,AveragePooling2D,Dense,Layer
from tensorflow.python.keras import layers,Model,Sequential
import numpy as np

print(tf.__version__)
class GeneratorLoss(Model):
    def __init__(self):
        super(GeneratorLoss, self).__init__()

        # 加载VGG16模型并提取前31层作为loss_network
        vgg = VGG16(weights='imagenet', include_top=False)
        loss_network = tf.keras.Sequential()
        for layer in vgg.layers[:31]:
            layer.trainable = False
            loss_network.add(layer)
        loss_network.trainable = False
        self.loss_network = loss_network

        # 定义MSE损失函数和TV Loss（总变差损失）,傅里叶高频损失
        self.mse_loss = MeanSquaredError()
        self.tv_loss = TVLoss(tv_loss_weight=1)
        self.fly_loss = FlyLoss()

    def call(self,out_images, target_images):

        # 对抗损失函数
        #adversarial_loss = tf.reduce_mean(1 - out_labels)

        # 感知损失函数
        perception_loss = self.mse_loss(self.loss_network(target_images), self.loss_network(out_images))

        # 图片损失函数
        image_loss = self.mse_loss(target_images, out_images)

        # TV Loss
        tv_loss = self.tv_loss(out_images)

        # 灰度图片的高频傅里叶损失
        fly_imag_loss= self.mse_loss(self.fly_loss(out_images),self.fly_loss(target_images))

        # 返回加权之后的损失值
        out = image_loss + 2e-7 * tv_loss + 0.5*fly_imag_loss + 0.0005 * perception_loss #+ 0.001 * adversarial_loss
        return out


class FlyLoss(Layer):
    def __init__(self):
        super(FlyLoss, self).__init__()

    def call(self, rgb_image):
        # 将彩色图像转换为灰度图像
        gray_images = tf.image.rgb_to_grayscale(rgb_image)
        gray_images = tf.squeeze(gray_images, axis=-1)

        # 进行二维傅里叶变换
        dft = tf.signal.fft2d(tf.cast(gray_images,tf.complex64))
        dft_shift = tf.signal.fftshift(dft)

        # 取对数后进行归一化
        # magnitude_spectrum_mod = 20 * tf.math.log(tf.abs(dft_shift) + 1)
        # magnitude_spectrum_mod /= tf.reduce_max(magnitude_spectrum_mod)
        #
        # magnitude_spectrum_real = 20 * tf.math.log(tf.abs(tf.math.real(dft_shift)) + 1)
        # magnitude_spectrum_real /= tf.reduce_max(magnitude_spectrum_real)

        magnitude_spectrum_imag = 20 * tf.math.log(tf.abs(tf.math.imag(dft_shift)) + 1)
        magnitude_spectrum_imag /= tf.reduce_max(magnitude_spectrum_imag)

        return magnitude_spectrum_imag


class TVLoss(Layer):
    def __init__(self, tv_loss_weight=1.0):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def call(self, x):
        batch_size = tf.shape(x)[0]
        h_x = tf.shape(x)[1]
        w_x = tf.shape(x)[2]
        count_h = tf.cast(self.tensor_size(x[:, 1:, :, :]), dtype=tf.float32)
        count_w = tf.cast(self.tensor_size(x[:, :, 1:, :]), dtype=tf.float32)
        h_tv = tf.reduce_sum(tf.pow((x[:, 1:, :, :] - x[:, :h_x - 1, :, :]), 2))
        w_tv = tf.reduce_sum(tf.pow((x[:, :, 1:, :] - x[:, :, :w_x - 1, :]), 2))
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / tf.cast(batch_size, dtype=tf.float32)

    @staticmethod
    def tensor_size(t):
        return tf.shape(t)[1] * tf.shape(t)[2] * tf.shape(t)[3]




if __name__ == '__main__':
    model = GeneratorLoss()
    np.random.seed(666)
    out_labels = tf.convert_to_tensor(np.random.normal(size=(64,1)).astype("float32"))
    out_images = tf.convert_to_tensor(np.random.normal(size=(64,32,32,3)).astype("float32"))
    target_images = tf.convert_to_tensor(np.random.normal(size=(64,32,32,3)).astype("float32"))

    y = model(out_labels,out_images,target_images)
    print(y.numpy())





