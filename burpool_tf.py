import tensorflow as tf
import cv2
class burpool(tf.keras.layers.Layer):
    def __init__(self, channels, pad_type="REFLECT", filt_size=4, stride=2, pad_off=0):
        super(burpool, self).__init__()
        self.channels = channels
        self.pad_type = pad_type
        self.filt_size = filt_size
        self.stride = stride
        self.pad_off = pad_off

    def build(self, input_shape):
        self.pad_sizes = [int(1. * (self.filt_size - 1) / 2), int(tf.math.ceil(1. * (self.filt_size - 1) / 2)),
                          int(1. * (self.filt_size - 1) / 2), int(tf.math.ceil(1. * (self.filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + self.pad_off for pad_size in self.pad_sizes]

        if (self.filt_size == 1):
            a = tf.convert_to_tensor([1., ])
        elif (self.filt_size == 2):
            a = tf.convert_to_tensor([1., 1.])
        elif (self.filt_size == 3):
            a = tf.convert_to_tensor([1., 2., 1.])
        elif (self.filt_size == 4):
            a = tf.convert_to_tensor([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = tf.convert_to_tensor([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = tf.convert_to_tensor([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = tf.convert_to_tensor([1., 6., 15., 20., 15., 6., 1.])
        filt = tf.convert_to_tensor(a[:, None] * a[None, :])
        filt = filt / tf.reduce_sum(filt)
        filt  = tf.expand_dims(tf.expand_dims(filt , axis=-1),axis=-1)
        self.filter = tf.tile(filt, [1, 1, 1, self.channels])
        self.pad= tf.pad
        super(burpool, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x =self.pad(inputs,[[0,0],[self.pad_sizes[0],self.pad_sizes[1]],[self.pad_sizes[2],self.pad_sizes[3]],[0,0]],mode=self.pad_type)

        return tf.nn.conv2d(x,self.filter,strides=self.stride,padding="VALID")

# model = burpool(channels=3,stride=2)
# img = cv2.imread(r"D:\pj\tools\data\1.png") / 255.0
# x = tf.expand_dims(img,axis=0)
# y = model(x)
#
# cv2.imshow("1",tf.squeeze(y,axis=0).numpy())
# cv2.waitKey(0)
# print(y.shape)
