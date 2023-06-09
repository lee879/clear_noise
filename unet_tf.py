from burpool_tf import burpool
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D,Dense,Conv2DTranspose,Activation

class block1(tf.keras.layers.Layer):
    def __init__(self):
        super(block1, self).__init__()
        self.block_1 = tf.keras.Sequential([
            Conv2D(48,3,1,padding="same",kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
            Conv2D(48,3,1,padding="same",kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
            burpool(channels=48),
        ])
    def call(self, inputs, *args, **kwargs):
        return self.block_1(inputs)

class block2(tf.keras.layers.Layer):
    def __init__(self):
        super(block2, self).__init__()

        self.block_2 = tf.keras.Sequential([
            Conv2D(48, 3, 1, padding="same", kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
            burpool(channels=48),
        ])
    def call(self, inputs, *args, **kwargs):
        return self.block_2(inputs)

class block3(tf.keras.layers.Layer):
    def __init__(self):
        super(block3, self).__init__()
        self.block_3 = tf.keras.Sequential([
            Conv2D(48, 3, 1, padding="same", kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
            Conv2DTranspose(48,3,2,padding="same",kernel_initializer="he_normal")
        ])
    def call(self, inputs, *args, **kwargs):
        return self.block_3(inputs)

class block4(tf.keras.layers.Layer):
    def __init__(self):
        super(block4, self).__init__()
        self.block_4 = tf.keras.Sequential([
            Conv2D(96,3,1,"same",kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
            Conv2D(96, 3, 1, "same", kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
            Conv2DTranspose(96,3,2,"same",kernel_initializer="he_normal")
        ])

    def call(self, inputs, *args, **kwargs):
        return self.block_4(inputs)

class block5(tf.keras.layers.Layer):
    def __init__(self,out_channels):
        super(block5, self).__init__()
        self.block_5 = tf.keras.Sequential([
            Conv2D(96, 3, 1, "same", kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
            Conv2D(64, 3, 1, "same", kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
            Conv2D(32, 3, 1, "same", kernel_initializer="he_normal"),
            Activation(tf.nn.relu),
            Conv2D(out_channels, 1, 1, "same", kernel_initializer="he_normal"),
        ])

    def call(self, inputs, *args, **kwargs):
        return self.block_5(inputs)

class unet(tf.keras.Model):
    def __init__(self,out_channels):
        super(unet, self).__init__()
        self.l1 = block1()
        self.l2 = [block2() for _ in range(4)]
        self.l3 = block3()
        self.l4 = [block4() for _ in range(4)]
        self.l5 = block5(out_channels=out_channels)

    def call(self, inputs, training=None, mask=None):
        # this Encoder
        x0 = self.l1(inputs)
        x1 = self.l2[0](x0)
        x2 = self.l2[1](x1)
        x3 = self.l2[2](x2)
        x4 = self.l2[3](x3)

        # this Decoder
        y1 = self.l3(x4)
        y1_cat = tf.concat([y1,x3],axis=-1)
        y2 = self.l4[0](y1_cat)
        y2_cat = tf.concat([y2,x2],axis=-1)
        y3 = self.l4[1](y2_cat)
        y3_cat = tf.concat([y3, x1], axis=-1)
        y4 = self.l4[2](y3_cat)
        y4_cat = tf.concat([y4, x0], axis=-1)
        y5 = self.l4[3](y4_cat)
        y5_cat = tf.concat([y5, inputs], axis=-1)
        return self.l5(y5_cat)

# model = unet(out_channels=12)
# x = tf.random.normal(shape=[32,512,512,12])
# y = model(x)
# model.summary()
# print(y.shape)





