import keras.backend as K
from keras.losses import mean_squared_error
from keras.models import Model
from keras.applications.vgg16 import VGG16

from toolbox.layers import VGGNormalize
from toolbox.models import vgg16


def mse(**kwargs):
    return 'mse'


def dummy(**kwargs):
    return lambda x, y: K.variable(0.0)


def tv(**kwargs):
    """Total variation
    Smooth the output image
    """
    weight = 1.0 if not 'weight' in kwargs else kwargs['weight']

    def loss(y_true, y_pred):
        x_out = y_pred
        shape = K.shape(x_out)
        img_width, img_height, channel = (shape[1], shape[2], shape[3])
        size = img_width * img_height * channel
        a = K.square(x_out[:, :img_width - 1, :img_height - 1, :] - x_out[:, 1:, :img_height - 1, :])
        b = K.square(x_out[:, :img_width - 1, :img_height - 1, :] - x_out[:, :img_width - 1, 1:, :])
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
        return weight * mean_squared_error(feature_true, feature_pred)

    return loss


def combined(weight1=1, weight2=1e-6, **kwargs):
    loss1 = pl(weight=weight1, **kwargs)
    loss2 = tv(weight=weight2, **kwargs)

    def loss(y_true, y_pred):
        return loss1(y_true, y_pred) + loss2(y_true, y_pred)

    return loss


def get_loss(name, **kwargs):
    # Default loss is mse
    if name is None:
        return mse()
    return globals()[name](**kwargs)
