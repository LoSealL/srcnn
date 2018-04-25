import keras.backend as K
from keras.losses import mean_squared_error
from keras.models import Model
from keras.applications.vgg16 import VGG16

from toolbox.layers import VGGNormalize


def mse(**kwargs):
    return 'mse'


def dummy(**kwargs):
    return lambda x, y: K.variable(0.0)


def mse_gray(**kwargs):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_true[..., :1] - y_pred[..., :1]))

    return loss


def tv(**kwargs):
    """Total variation
    Smooth the output image
    """
    weight = 1.0 if not 'weight' in kwargs else kwargs['weight']

    def loss(y_true, y_pred):
        x_out = y_pred
        shape = K.shape(x_out)
        img_width, img_height, channel = (shape[1], shape[2], shape[3])
        channel = K.minimum(3, channel)
        size = img_width * img_height * channel
        a = K.square(x_out[:, :img_width - 1, :img_height - 1, :3] - x_out[:, 1:, :img_height - 1, :3])
        b = K.square(x_out[:, :img_width - 1, :img_height - 1, :3] - x_out[:, :img_width - 1, 1:, :3])
        return weight * K.sum(K.pow(a + b, 1.25)) / K.cast(size, K.floatx())

    return loss


def pl(block=2, conv=2, **kwargs):
    """Perceptual Loss Function

    paper: https://arxiv.org/abs/1603.08155
    """
    layer_name = f'block{block}_conv{conv}'
    vgg = VGG16(input_shape=[None, None, 3], include_top=False)
    inp = vgg.input
    outp = vgg.get_layer(layer_name).output
    model = Model(inputs=inp, outputs=outp, name='perceptual_loss')
    weight = 1.0 if not 'weight' in kwargs else kwargs['weight']

    def loss(y_true, y_pred):
        y_true.set_shape(y_pred.shape)
        y_true_norm = VGGNormalize()(y_true)
        y_pred_norm = VGGNormalize()(y_pred)
        feature_true = model(y_true_norm)
        feature_pred = model(y_pred_norm)
        return weight * K.mean(K.square(feature_true - feature_pred))

    return loss


def mse_pl_tv(weight0=1, weight1=4, weight2=1e-6, **kwargs):
    loss1 = pl(weight=weight1, **kwargs)
    loss2 = tv(weight=weight2, **kwargs)

    def loss(y_true, y_pred):
        l_mse = weight0 * K.mean(K.square(y_true - y_pred))
        l_pl = loss1(y_true, y_pred)
        l_tv = loss2(y_true, y_pred)
        return l_mse + l_pl + l_tv

    return loss


def get_loss(name, **kwargs):
    # Default loss is mse
    if name is None:
        return mse()
    return globals()[name](**kwargs)
