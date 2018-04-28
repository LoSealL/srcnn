from keras.layers import *
from keras.regularizers import l2
from keras.models import Sequential, Model

from toolbox.layers import *


def bicubic(c, scale=3):
    model = Sequential()
    model.add(InputLayer(input_shape=[None, None, c], dtype='uint8', name='input_lr'))
    model.add(PreProcess())
    model.add(ImageRescale(scale))
    model.add(PostProcess(c))
    return model


def composeModel(c, scale, model, bgr=False):
    """Convert input RGB image to grayscale one and
    concatenate UV channel after prediction

    """

    def slice(x, a, b):
        return x[..., a:b]

    rgb = Input(shape=[None, None, c], dtype='uint8', name='input_lr_rgb')
    frgb = PreProcess()(rgb)
    frgb = Scale(1 / 255.0)(frgb)
    yuv = RGB2YUV(bgr=bgr)(frgb)
    y = Lambda(slice, arguments={'a': 0, 'b': 1})(yuv)
    uv = Lambda(slice, arguments={'a': 1, 'b': 3})(yuv)
    uv_scaled = ImageRescale(scale)(uv)
    y = Scale(255.0)(y)
    y_pred = model(y)
    y_pred = Scale(1 / 255.0)(y_pred)
    yuv_scaled = Concatenate()([y_pred, uv_scaled])
    rgb_scaled = YUV2RGB(bgr=bgr)(yuv_scaled)
    rgb_scaled = PostProcess(c)(Scale(255.0)(rgb_scaled))
    model = Model(rgb, rgb_scaled)
    return model


def srcnn(c, f=(9, 1, 5), n=(64, 32), scale=3):
    """Build an SRCNN model.

    See https://arxiv.org/abs/1501.00092
    """
    assert len(f) == len(n) + 1
    model = Sequential()
    model.add(InputLayer([None, None, c], dtype='uint8', name='input_lr'))
    model.add(PreProcess())
    model.add(ImageRescale(scale))
    channel = min(3, c)
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2D(channel, f[-1], padding='same',
                     kernel_initializer='he_normal'))
    model.add(PostProcess(c))
    return model


def fsrcnn(c, d=56, s=12, m=4, scale=3):
    """Build an FSRCNN model.

    See https://arxiv.org/abs/1608.00367
    """
    model = Sequential()
    model.add(InputLayer([None, None, c], dtype='uint8', name='input_lr'))
    model.add(PreProcess())
    f = [5, 1] + [3] * m + [1]
    n = [d, s] + [s] * m + [d]
    channel = min(3, c)
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2DTranspose(channel, 9, strides=scale, padding='same',
                              kernel_initializer='he_normal'))
    model.add(PostProcess(c))
    return model


def nsfsrcnn(c, d=56, s=12, m=4, scale=3, pos=1):
    """Build an FSRCNN model, but change deconv position.

    See https://arxiv.org/abs/1608.00367
    """
    model = Sequential()
    model.add(InputLayer([None, None, c], dtype='uint8', name='input_lr'))
    model.add(PreProcess())
    f1 = [5, 1] + [3] * pos
    n1 = [d, s] + [s] * pos
    f2 = [3] * (m - pos - 1) + [1]
    n2 = [s] * (m - pos - 1) + [d]
    f3 = 9
    channel = min(3, c)
    for ni, fi in zip(n1, f1):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2DTranspose(s, 3, strides=scale, padding='same',
                              kernel_initializer='he_normal'))
    for ni, fi in zip(n2, f2):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2D(channel, f3, padding='same',
                     kernel_initializer='he_normal'))
    model.add(PostProcess(c))
    return model


def espcn(c, f=(5, 3, 3), n=(64, 32), scale=3):
    """Build an ESPCN model.

    See https://arxiv.org/abs/1609.05158
    """

    assert len(f) == len(n) + 1
    model = Sequential()
    model.add(InputLayer([None, None, c], dtype='uint8', name='input_lr'))
    model.add(PreProcess())
    channel = min(3, c)
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='tanh'))
    model.add(Conv2D(channel * scale ** 2, f[-1], padding='same',
                     kernel_initializer='he_normal'))
    model.add(Conv2DSubPixel(scale, channel))
    model.add(PostProcess(c))
    return model


def espcn_rgb(c, f=(5, 3, 3), n=(64, 32), scale=3):
    """Build an ESPCN model.

    See https://arxiv.org/abs/1609.05158
    """

    def slice(x, a, b):
        return x[..., a:b]

    assert len(f) == len(n) + 1
    rgb = Input([None, None, c], dtype='uint8', name='input_lr')
    rgb_float = PreProcess()(rgb)
    rgb_float = Scale(1 / 255.0)(rgb_float)
    yuv = RGB2YUV()(rgb_float)
    y = Lambda(slice, arguments={'a': 0, 'b': 1})(yuv)
    uv = Lambda(slice, arguments={'a': 1, 'b': 3})(yuv)
    uv_scaled = ImageRescale(scale)(uv)
    for ni, fi in zip(n, f):
        y = Conv2D(ni, fi, padding='same',
                   kernel_initializer='he_normal', activation='tanh')(y)
    y = Conv2D(scale ** 2, f[-1], padding='same', kernel_initializer='he_normal')(y)
    y_scaled = Conv2DSubPixel(scale, 1)(y)
    yuv_scaled = Concatenate()([y_scaled, uv_scaled])
    rgb_scaled = YUV2RGB()(yuv_scaled)
    rgb_scaled = PostProcess(c)(Scale(255.0)(rgb_scaled))
    model = Model(rgb, rgb_scaled, name='ESPCN_RGB')
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
        # t = Dropout(dropout)(t)
        return t

    inputs = Input([None, None, c], dtype='uint8', name='input_lr')
    xflt = PreProcess()(inputs)
    xbic = ImageRescale(scale)(xflt)
    x = [xflt]
    channel = min(3, c)
    for i in range(nfilters):
        x.append(conv2d(x[i], f[i], 3))
    x_concat = Concatenate(axis=3)(x[1:])
    a1 = conv2d(x_concat, A, 1, name='A1')
    b1 = conv2d(x_concat, B[0], 1, name='B1')
    b2 = conv2d(b1, B[1], 3, name='B2')
    x_concat = Concatenate(axis=3)([a1, b2])
    x_out = conv2d(x_concat, scale ** 2, 1)
    x_out = Conv2DSubPixel(scale, channel)(x_out)
    y = Add()([x_out, xbic])
    x_out = PostProcess(c)(x_out)
    model = Model(inputs, x_out, name='DCSCN')
    return model


def dcscn_rgb(c, nfilters=7, f=(96, 76, 65, 55, 47, 39, 32), A=64, B=(32, 32), scale=3):
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

    def slice(x, a, b):
        return x[..., a:b]

    assert c >= 3
    rgb = Input([None, None, c], dtype='uint8', name='input_lr')
    rgb_float = PreProcess()(rgb)
    rgb_float = Scale(1 / 255.0)(rgb_float)
    yuv = RGB2YUV()(rgb_float)
    y = Lambda(slice, arguments={'a': 0, 'b': 1})(yuv)
    uv = Lambda(slice, arguments={'a': 1, 'b': 3})(yuv)
    uv_scaled = ImageRescale(scale)(uv)
    x = [y]
    for i in range(nfilters):
        x.append(conv2d(x[i], f[i], 3))
    x_concat = Concatenate(axis=3)(x[1:])
    a1 = conv2d(x_concat, A, 1, name='A1')
    b1 = conv2d(x_concat, B[0], 1, name='B1')
    b2 = conv2d(b1, B[1], 3, name='B2')
    x_concat = Concatenate(axis=3)([a1, b2])
    x_out = conv2d(x_concat, scale ** 2, 1)
    x_out = Conv2DSubPixel(scale, 1)(x_out)
    yuv_scaled = Concatenate()([x_out, uv_scaled])
    rgb_scaled = YUV2RGB()(yuv_scaled)
    rgb_scaled = PostProcess(c)(Scale(255.0)(rgb_scaled))
    model = Model(rgb, rgb_scaled, name='DCSCN')
    return model


def red(c, depth, f, k=3, stride=None, skip_step=2, scale=3):
    """Residual Encoder Decoder Network

    See https://arxiv.org/abs/1603.09056
    """

    inputs = Input([None, None, c], dtype='uint8')
    inp = CastUInt2Float()(inputs)
    inp = ImageRescale(scale)(inp)
    assert (depth % 2 == 0)
    stride = [1] * (depth // 2) if not stride else stride
    assert (len(stride) == depth // 2)
    # Encoder convolution
    enc = [inp]
    skip = [True] if skip_step else None
    for i in range(depth // 2):
        x = Conv2D(f, k, strides=stride[i], padding='same',
                   activation='relu', kernel_initializer='he_normal')(enc[i])
        enc += [x]
        if skip: skip += [(i + 1) % skip_step == 0]
    if skip: skip[-1] = False
    # Decoder deconvolution add skip connection
    if skip: skip.reverse()
    enc.reverse()
    stride.reverse()
    dec = [enc[0]]
    for i in range(depth // 2):
        if skip and skip[i]:
            dec[i] = Add()([dec[i], enc[i]])
            dec[i] = ThresholdedReLU(theta=0.0)(dec[i])
        f = c if i + 1 == depth // 2 else f
        x = Conv2DTranspose(f, k, strides=stride[i], padding='same',
                            activation='relu', kernel_initializer='he_normal')(dec[i])
        dec += [x]
    # output
    if skip:
        outputs = Add()([dec[-1], enc[-1]])
    else:
        outputs = dec[-1]
    model = Model(inputs, outputs, name="RED")
    return model


def flow_estimate(c, k, n, s, name=None):
    """optical flow motion estimation

    :param c: scalar, input channel
    :param k: tuple, kernel sizes
    :param n: tuple, filter numbers
    :param s: tuple, strides
    :return: ME model
    """

    from keras.initializers import Orthogonal
    inp = Input(shape=[None, None, c])
    nn = []
    orth = Orthogonal(np.sqrt(2))
    for _k, _n, _s in zip(k, n, s):
        nn.append(Conv2D(_n, _k, strides=_s, padding='same', activation='relu', kernel_initializer=orth))
    nn[-1] = Conv2D(n[-1], k[-1], strides=s[-1], padding='same', activation='tanh', kernel_initializer=orth)
    x = inp
    for _nn in nn: x = _nn(x)
    scale = 1
    for _s in s: scale *= _s
    output = Conv2DSubPixel(scale, 2)(x)
    return Model(inp, output, name=name)


def coarse_flow(c=2, k=(5, 3, 5, 3, 3), n=(24, 24, 24, 24, 32), s=(2, 1, 2, 1, 1)):
    return flow_estimate(c, k, n, s, name='CoarseFlowEstimation')


def fine_flow(c=5, k=(5, 3, 3, 3, 3), n=(24, 24, 24, 24, 8), s=(2, 1, 1, 1, 1)):
    return flow_estimate(c, k, n, s, name='FineFlowEstimation')


def spmc():
    """Spatial transformer motion compensation

    :see ...
    """

    inp = Input(shape=[2, None, None, 1])
    x = Lambda(lambda t: K.concatenate([t[:, 0], t[:, 1]]))(inp)
    coarse_me = coarse_flow()(x)
    x = Lambda(lambda t: K.concatenate([t[0][:,1], t[1]]))([inp, coarse_me])
    i_coarse = MotionCompensation(name='CoarseWarp')(x)
    x = Lambda(lambda t: K.concatenate([t[0][:, 0], t[0][:, 1], t[1], t[2]]))([inp, coarse_me, i_coarse])
    fine_me = fine_flow()(x)
    total_me = Add()([coarse_me, fine_me])
    x = Lambda(lambda t: K.concatenate([t[0][:, 1], t[1]]))([inp, total_me])
    i_fine = MotionCompensation(name='FineWarp')(x)
    return Model(inp, i_fine, name='SPMC')


def get_model(name):
    return globals()[name]
