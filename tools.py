import numpy as np
import tensorflow as tf

def restore_patches_to_image(patches, patch_size):
    """
    :param patches: Patches of image (patch_num, C, win, win)
    :param patch_size: int
    :return: Restored image (C * W * H) Numpy
    """
    patches = np.transpose(patches, [0,3, 2, 1])

    patch_num, C, _, _ = patches.shape
    W = patch_size * int(np.sqrt(patch_num))
    H = patch_size * int(np.sqrt(patch_num))

    image = np.zeros((C, W, H), dtype=np.float32)
    idx = 0
    for ws in range(0, W, patch_size):
        for hs in range(0, H, patch_size):
            patch = patches[idx]
            image[:, ws:ws + patch_size, hs:hs + patch_size] = patch
            idx += 1

    return np.transpose(image, [2, 1, 0])

# x = np.random.random(size=(1024,32,32,3))
# img = restore_patches_to_image(x,32)



def image_to_patches(image, patch_size):
    """
    :param image: Image (C * W * H) Numpy
    :param patch_size: int
    :return: (patch_num, C, win, win)
    """
    image = np.transpose(image, [2, 1, 0])
    W = image.shape[1]
    H = image.shape[2]
    if W < patch_size or H < patch_size:
        return []

    ret = []
    for ws in range(0, W // patch_size):
        for hs in range(0, H // patch_size):
            patch = image[:, ws * patch_size : (ws + 1) * patch_size, hs * patch_size : (hs + 1) * patch_size]
            ret.append(np.transpose(patch, [2, 1, 0]))
    return np.array(ret, dtype=np.float32)

def images_to_patches(images, patch_size):
    """
    :param images: List[Image (C * W * H)]
    :param patch_size: int
    :return: (n * C * W * H)
    """
    patches_list = []
    for image in images:
        patches = image_to_patches(image, patch_size=patch_size)
        if len(patches) != 0:
            patches_list.append(patches)
    del images
    return np.vstack(patches_list)


def downsample(x):

    x = np.transpose(x, [0, 3, 2, 1])
    N, C, W, H = x.shape
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    Cout = 4 * C
    Wout = W // 2
    Hout = H // 2

    down_features = np.zeros((N, Cout, Wout, Hout))

    for idx in range(4):
        down_features[:, idx:Cout:4, :, :] = x[:, :, idxL[idx][0]::2, idxL[idx][1]::2]

    return np.transpose(down_features, [0, 3, 2, 1])

def upsample(x):

    x = np.transpose(x,[0,3,2,1])
    N, Cin, Win, Hin = x.shape
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    Cout = Cin // 4
    Wout = Win * 2
    Hout = Hin * 2

    up_feature = np.zeros((N, Cout, Wout, Hout))
    for idx in range(4):
        up_feature[:, :, idxL[idx][0]::2, idxL[idx][1]::2] = x[:, idx:Cin:4, :, :]

    return np.transpose(up_feature, [0, 3, 2, 1])

# x = np.random.normal(size=(8,64,64,3))
# print(x.shape)
# y = downsample(x)
# print(y.shape)
# y = upsample(y)
# print(y.shape)