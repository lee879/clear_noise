import numpy as np
import cv2
import tensorflow as tf
from skimage.util import random_noise

def add_gaussian_noise(image,variance):
    '''
    :param image:
    :param mean:
    :param variance:
    :return: 高斯彩色噪声
    '''
    noisy_image = random_noise(image,var=variance)
    return noisy_image

def add_salt_and_pepper_noise_0(image, density):
    '''
    :param image:
    :param density:
    :return: 均匀彩色椒盐噪声
    '''
    image = np.array(image).astype(np.float32)
    salt_pepper = tf.random.uniform(shape=tf.shape(image), minval=0, maxval=1, dtype=tf.float32)
    salt = tf.less(salt_pepper, density / 2)
    pepper = tf.less(salt_pepper, density) & tf.logical_not(salt)
    noisy_image = tf.where(salt, tf.ones_like(image) * 255, image)
    noisy_image = tf.where(pepper, tf.zeros_like(image), noisy_image)
    noisy_image = tf.cast(noisy_image, tf.uint8)
    return noisy_image


def add_bw_noise(img_batch, density=0.2):
    '''
    :param img_batch:
    :param density:
    :return: 均匀黑白椒盐噪声（不是给的高斯）
    '''
    assert img_batch.ndim == 4, "输入应该是一个四维数组，代表图像集合"

    # 创建一个副本，避免修改原始数据
    img_batch_noisy = np.copy(img_batch)

    noise = np.zeros(shape=(img_batch.shape[0], img_batch.shape[1], img_batch.shape[2]), dtype=np.int32)
    noise = np.random.uniform(0, 1, noise.shape)
    threshold = 1 - density / 2
    noise[noise > threshold] = 255
    noise[noise < density] = -255
    k = noise

    k1 = k == 255
    k1 = np.int32(k1)
    K_1 = np.stack((k1, k1, k1), axis=-1) * 255
    img_batch_noisy[(img_batch_noisy + K_1) >= 255] = 255

    k2 = k == -255
    k2 = np.int32(k2)
    K_2 = np.stack((k2, k2, k2), axis=-1) * -255
    img_batch_noisy[(img_batch_noisy + K_2) <= 0] = 0

    return img_batch_noisy

def add_blur(img_batch, kernel_size=(0, 0), sigma=0):
    '''
    :param img_batch:
    :param kernel_size:
    :param sigma:
    :return: 高斯模糊
    '''
    assert img_batch.ndim == 4, "输入应该是一个四维数组，代表图像集合"
    blurred_images = np.zeros_like(img_batch)
    for i in range(img_batch.shape[0]):
        blurred_images[i] = cv2.GaussianBlur(img_batch[i], kernel_size, sigmaX=sigma)
    return blurred_images

def add_distort(images, distortion_prob=0.5, distortion_scale=0.2):
    '''
    :param images:
    :param distortion_prob:
    :param distortion_scale:
    :return: 批量产生不同的畸变图片的函数。
    '''
    distorted_images = []
    for image in images:
        distorted_image = image.copy()
        if np.random.uniform() < distortion_prob:
            # 随机选择畸变类型
            distortion_type = np.random.choice(['brightness', 'rotation',"dis"])
            if distortion_type == 'brightness':
                # 亮度畸变
                brightness_scale = 1 + np.random.uniform(-distortion_scale, distortion_scale)
                distorted_image = cv2.convertScaleAbs(distorted_image, alpha=brightness_scale)
            elif distortion_type == 'rotation':
                # 旋转畸变
                angle = np.random.uniform(-distortion_scale * 180, distortion_scale * 180)
                rows, cols = image.shape[:2]
                rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                distorted_image = cv2.warpAffine(distorted_image, rotation_matrix, (cols, rows))
            elif distortion_type == 'dis':
                # 获取图像尺寸
                height, width, channels = image.shape

                # 随机生成畸变参数
                distortion = np.random.uniform(-distortion_scale, distortion_scale, size=(4,))

                # 定义畸变矩阵
                distortion_matrix = np.array([[1 + distortion[0], distortion[1], distortion[2]],
                                            [distortion[1], 1 + distortion[3], 0]])

                # 进行图像畸变处理
                distorted_image = cv2.warpAffine(image, distortion_matrix, (width, height))
        distorted_images.append(distorted_image)
    return np.array(distorted_images).astype(np.uint8)


def gaussian_noise(img):
    noise_img = img.astype(np.float32)
    stddev = np.random.uniform(0, 50)
    noise = np.random.randn(*img.shape) * stddev
    noise_img += noise
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    return noise_img

def add_gaussian_noise(image, mean, stddev):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def add_salt_and_pepper_noise(image, salt_ratio, pepper_ratio):
    noisy_image = np.copy(image)
    num_salt = int(image.size * salt_ratio)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords] = 255

    num_pepper = int(image.size * pepper_ratio)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords] = 0

    return noisy_image

def add_multiplicative_noise(image, noise_factor):
    noise = np.random.uniform(1 - noise_factor, 1 + noise_factor, image.shape)
    noisy_image = image.astype(np.float32) * noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image
# scal = 1024
# img = cv2.imread(r"D:\pj\FFDNet_pytorch-master\FfdNet\data\image\6.png") / 255.0
# img = cv2.resize(img,(int(img.shape[1] * (scal / img.shape[0])),scal))
# cv2.imshow("1",img)
# Img = np.expand_dims(img,0)
# img = add_gaussian_noise(Img,variance=0/255)
# img1 = np.squeeze(img,0)
# cv2.imshow("2",img1)
#
# # img = add_bw_noise(Img,density=0.1)
# # img2 = np.squeeze(img,0)
# # cv2.imshow("3",img2)
# #
# # img = add_gaussian_noise(Img,variance=0.1)
# # img3 = np.squeeze(img,0)
# # cv2.imshow("4",img3)
#
# cv2.waitKey(0)