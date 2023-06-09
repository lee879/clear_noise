import os
from unet_tf import unet
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
import numpy as np
import glob
from dataset import make_dataset
from tools import upsample,downsample,images_to_patches
from noise import add_blur,add_gaussian_noise,gaussian_noise
import cv2
from loss import GeneratorLoss

ckpt_path = r"D:\pj\1\ckpt"
log = r"D:\pj\1\log"
batch = 2
img_size = 1024
epochs = 100000000
patch_size = 128
lr = 1e-5

def psnr(img1, img2):
    # 将图像数据类型转换为浮点数
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 计算图像的均方误差（MSE）
    mse = np.mean((img1 - img2) ** 2)

    # 计算PSNR
    if mse == 0:
        return float('inf')
    else:
        max_pixel = np.max(img1)
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

def g_loss_fn(model,loss_model,input_img,tar_img):
        y_pred = model(input_img)
        loss = loss_model(y_pred,tar_img)
        #loss = tf.math.pow(tf.math.abs(tar_img-y_pred),2.0)
        return loss,y_pred

def main():
    unt_model = unet(out_channels=3)
    loss_model = GeneratorLoss()
    unt_model.load_weights("./ckpt/best_d.hd5")

    best_weights_checkpoint_path_g = os.path.join(ckpt_path, 'best_d.hd5')
    best_weights_checkpoint_g = ModelCheckpoint(best_weights_checkpoint_path_g,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode='min')
    # 添加模糊
    # noise_scal = [0,20,1]
    # noise_scal = np.array([i for i in np.arange(noise_scal[0], noise_scal[1], noise_scal[2])])

    #建立log_path
    summary_writer = tf.summary.create_file_writer(log)

    # dataset
    img_path = glob.glob(r"D:\pj\FFDNet_pytorch-master\FfdNet\data\4\*.*")
    dataset, img_shape, _ = make_dataset(img_path, batch, shuffle=batch * 8, resize=img_size)
    dataset = dataset.repeat()
    db_iter = iter(dataset)
    img_noise = []
    img_noise_target = []

    for epoch in range(epochs):
        img = np.array(next(db_iter),dtype="float32")                               # 获得图片数据
        img_patch = images_to_patches(img,patch_size)
        for i in range(img_patch.shape[0]):
            img_noise.append(gaussian_noise(img_patch[i]))
        img_noise = np.array(img_noise) / 255. # input 数据
        #img_noise_input = downsample(img_noise)

        #img_noise_target.append(img[0])  # 加上一张原图片
        for i in range(img_patch.shape[0]):
            #img_noise_target.append(gaussian_noise(img_patch[i]))
            img_noise_target.append(img_patch[i])
        img_noise_target = np.array(img_noise_target,dtype="float32") / 255  #loss 数据
        #img_noise_target_input = tf.convert_to_tensor(downsample(img_noise_target),dtype=tf.float32)

        with tf.GradientTape() as tape:
            loss,y_p = g_loss_fn(unt_model,loss_model,img_noise,img_noise_target)

        grads = tape.gradient(loss, unt_model.trainable_variables)
        tf.optimizers.Adam(learning_rate=lr).apply_gradients(zip(grads, unt_model.trainable_variables))

        #Y_P = np.array(upsample(y_p))
        #loss = np.mean(loss.numpy())
        with summary_writer.as_default():
            tf.summary.scalar('loss', float(loss), step=epoch)
            #tf.summary.scalar('PSNR', float(PSNR), step=epoch)
            if epoch % 100 == 0:
                PSNR = psnr(np.array(y_p), img_noise_target)
                print("epoch:", int(epoch), "loss", float(loss), "PSNR", float(PSNR))
                tf.summary.image("predict_img", y_p[:3], step=epoch)
                tf.summary.image("real_image", img/255., step=epoch)
                tf.summary.image("noise_target_image", img_noise_target[:3], step=epoch)
                tf.summary.image("noise_image", img_noise[:3] , step=epoch)
                print("tf.keras.backend.clear_session")
                unt_model.save_weights(best_weights_checkpoint_path_g)
                tf.keras.backend.clear_session()

        img_noise = []
        img_noise_target = []
    return None

if __name__ == '__main__':
    main()


