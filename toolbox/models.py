from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import InputLayer
from keras.models import Sequential
import tensorflow as tf

from toolbox.layers import ImageRescale
from toolbox.layers import Conv2DSubPixel


def bicubic(c, scale=3):
    model = Sequential()
    model.add(InputLayer(input_shape=[None, None, c]))
    model.add(ImageRescale(scale, method=tf.image.ResizeMethod.BICUBIC))
    return model


def srcnn(c, f=(9, 1, 5), n=(64, 32), scale=3):
    """Build an SRCNN model.

    See https://arxiv.org/abs/1501.00092
    """
    assert len(f) == len(n) + 1
    model = Sequential()
    model.add(InputLayer([None, None, c]))
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2D(c, f[-1], padding='same',
                     kernel_initializer='he_normal'))
    return model


def fsrcnn(c, d=56, s=12, m=4, scale=3):
    """Build an FSRCNN model.

    See https://arxiv.org/abs/1608.00367
    """
    model = Sequential()
    model.add(InputLayer(input_shape=[None, None, c]))
    f = [5, 1] + [3] * m + [1]
    n = [d, s] + [s] * m + [d]
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2DTranspose(c, 9, strides=scale, padding='same',
                              kernel_initializer='he_normal'))
    return model


def nsfsrcnn(c, d=56, s=12, m=4, scale=3, pos=1):
    """Build an FSRCNN model, but change deconv position.

    See https://arxiv.org/abs/1608.00367
    """
    model = Sequential()
    model.add(InputLayer(input_shape=[None, None, c]))
    f1 = [5, 1] + [3] * pos
    n1 = [d, s] + [s] * pos
    f2 = [3] * (m - pos - 1) + [1]
    n2 = [s] * (m - pos - 1) + [d]
    f3 = 9
    n3 = c
    for ni, fi in zip(n1, f1):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2DTranspose(s, 3, strides=scale, padding='same',
                              kernel_initializer='he_normal'))
    for ni, fi in zip(n2, f2):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2D(n3, f3, padding='same',
                     kernel_initializer='he_normal'))
    return model


def espcn(c, f=(5, 3, 3), n=(64, 32), scale=3):
    """Build an ESPCN model.

    See https://arxiv.org/abs/1609.05158
    """

    def ps_init(shape, dtype=None):
        r, _, C, _ = shape
        indices = []
        for i in range(r):
            indices.append([])
            for j in range(r):
                indices[i].append([])
                for _ in range(C):
                    indices[i][j].append(i * r + j)
        kernel = tf.one_hot(indices, r * r * C, dtype=dtype, name='ps_kernel')
        assert kernel.shape == shape
        return kernel

    assert len(f) == len(n) + 1
    model = Sequential()
    model.add(InputLayer(input_shape=[None, None, c]))
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='tanh'))
    model.add(Conv2D(c * scale ** 2, f[-1], padding='same',
                     kernel_initializer='he_normal'))
    # model.add(Conv2DTranspose(c, scale,
    #                           strides=scale,
    #                           padding='same',
    #                           kernel_initializer=ps_init,
    #                           trainable=False))
    model.add(Conv2DSubPixel(scale))
    return model


def vgg16(include_top=False, input_shape=None, input_tensor=None):
    from keras.layers import Input
    from keras.models import Model
    from keras.layers import MaxPooling2D, Conv2D
    from keras.engine.topology import get_source_inputs
    from keras.utils.data_utils import get_file
    import keras.backend as K
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    # Determine proper input shape
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)

    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models',
                            file_hash='6d6bbae143d832006294945121d1f1fc')
    model.load_weights(weights_path, True)
    return model


def get_model(name):
    return globals()[name]
