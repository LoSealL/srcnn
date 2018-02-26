from keras.layers import (
    Conv2D, Conv2DTranspose, SeparableConv2D,
    InputLayer, Input, LeakyReLU, Dropout, BatchNormalization as BN,
    Add, Concatenate
)
from keras.regularizers import l2
from keras.models import Sequential, Model

from toolbox.layers import (
    ImageRescale,
    Conv2DSubPixel,
    CastUInt2Float,
)


def bicubic(c, scale=3):
    model = Sequential()
    model.add(InputLayer(input_shape=[None, None, c], dtype='uint8'))
    model.add(CastUInt2Float())
    model.add(ImageRescale(scale))
    return model


def srcnn(c, f=(9, 1, 5), n=(64, 32), scale=3):
    """Build an SRCNN model.

    See https://arxiv.org/abs/1501.00092
    """
    assert len(f) == len(n) + 1
    model = Sequential()
    model.add(InputLayer([None, None, c], dtype='uint8'))
    model.add(CastUInt2Float())
    model.add(ImageRescale(scale))
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
    model.add(InputLayer([None, None, c], dtype='uint8'))
    model.add(CastUInt2Float())
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
    model.add(InputLayer([None, None, c], dtype='uint8'))
    model.add(CastUInt2Float())
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

    assert len(f) == len(n) + 1
    model = Sequential()
    model.add(InputLayer([None, None, c], dtype='uint8'))
    model.add(CastUInt2Float())
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='tanh'))
    model.add(Conv2D(c * scale ** 2, f[-1], padding='same',
                     kernel_initializer='he_normal'))
    model.add(Conv2DSubPixel(scale, c))
    return model


def drcn(c, f=256, k=3, scale=3):
    """Deeply-Recursive Convolutional Network

     See https://arxiv.org/abs/1511.04491
    """
    raise RuntimeError('Unimplemented!')


def fastsr(c, scale=3):
    model = Sequential()
    model.add(InputLayer([None, None, c], dtype='uint8'))
    model.add(CastUInt2Float())
    model.add(SeparableConv2D(64, 9, padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(SeparableConv2D(32, 5, padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(SeparableConv2D(c * scale ** 2, 5, padding='same',
                              activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2DSubPixel(scale, c))
    return model


def dcscn(c, nfilters=7, f=(96, 76, 65, 55, 47, 39, 32), A=64, B=(32, 32), scale=3):
    """Deep CNN with Skip Connection and Network in Network

    See https://arxiv.org/abs/1707.05425
    """

    def conv2d(t, f, k, l2_weight=0.0001, dropout=0.8, **kwargs):
        t = Conv2D(f, k, padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_weight),
                   **kwargs)(t)
        # t = PReLU()(t)
        # t = Dropout(dropout)(t)
        return t

    inputs = Input([None, None, c], dtype='uint8')
    xflt = CastUInt2Float()(inputs)
    xbic = ImageRescale(scale)(xflt)
    x = [xflt]
    for i in range(nfilters):
        x.append(conv2d(x[i], f[i], 3))
    x_concat = Concatenate(axis=3)(x[1:])
    a1 = conv2d(x_concat, A, 1, name='A1')
    b1 = conv2d(x_concat, B[0], 1, name='B1')
    b2 = conv2d(b1, B[1], 3, name='B2')
    x_concat = Concatenate(axis=3)([a1, b2])
    x_out = conv2d(x_concat, scale ** 2, 1)
    x_out = Conv2DSubPixel(scale, c)(x_out)
    y = Add()([x_out, xbic])
    model = Model(inputs, x_out, name='DCSCN')
    return model


def drrn(c, B=1, U=3, scale=3):
    """Deep Recursive Residual Network

    See http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf
    """

    from toolbox.layers import DRRNResidualBlock

    inputs = Input([None, None, c], dtype='uint8')
    inp = CastUInt2Float()(inputs)
    inp = ImageRescale(scale)(inp)
    x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(inp)
    for _ in range(B):
        x = DRRNResidualBlock(U, 128, 3)(x)
    x = Conv2D(c, 1, padding='same',
               activation='relu', kernel_initializer='he_normal')(x)
    outputs = Add()([x, inp])
    model = Model(inputs, outputs, name='DRRN')
    return model


def get_model(name):
    return globals()[name]
