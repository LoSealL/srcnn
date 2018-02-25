from keras.engine.topology import Layer
import keras.backend as K
import numpy as np
import tensorflow as tf

custom_layers = {}


class ImageRescale(Layer):
    def __init__(self, scale, method=tf.image.ResizeMethod.BICUBIC,
                 trainable=False, **kwargs):
        self.scale = scale
        self.method = method
        super().__init__(trainable=trainable, **kwargs)

    def compute_size(self, shape):
        size = np.array(shape)[[1, 2]] * self.scale
        return tuple(size.astype(int))

    def call(self, x, **kwargs):
        shape = K.shape(x)
        h, w, c = shape[1], shape[2], shape[3]
        return tf.image.resize_images(x, [h * self.scale, w * self.scale], method=self.method)

    def compute_output_shape(self, input_shape):
        h, w = input_shape[1:3]
        if w: w = w * self.scale
        if h: h = h * self.scale
        return input_shape[0], h, w, input_shape[-1]

    def get_config(self):
        config = super().get_config()
        config['scale'] = self.scale
        config['method'] = self.method
        return config


custom_layers['ImageRescale'] = ImageRescale


class Conv2DSubPixel(Layer):
    """Sub-pixel convolution layer.

    See https://arxiv.org/abs/1609.05158
    """

    def __init__(self, scale, channel=3, **kwargs):
        self.scale = scale
        self.channel = channel
        super().__init__(trainable=False, **kwargs)

    def call(self, t, **kwargs):
        r = self.scale
        shape = K.shape(t)
        H, W = shape[1], shape[2]
        C = self.channel
        t = K.reshape(t, [-1, H, W, r, r, C])
        # Here we are different from Equation 4 from the paper. That equation
        # is equivalent to switching 3 and 4 in `perm`. But I feel my
        # implementation is more natural.
        t = tf.transpose(t, perm=[0, 1, 3, 2, 4, 5])  # S, H, r, H, r, C
        t = K.reshape(t, [-1, H * r, W * r, C])
        return t

    def compute_output_shape(self, input_shape):
        r = self.scale
        H, W, rrC = np.array(input_shape[1:])
        H = H * r if H else None
        W = W * r if W else None
        C = rrC // (r ** 2) if rrC else None
        return input_shape[0], H, W, C

    def get_config(self):
        config = super().get_config()
        config['scale'] = self.scale
        return config


custom_layers['Conv2DSubPixel'] = Conv2DSubPixel


class Gray2RGB(Layer):
    """Convert grayscale image to RGB format

    """

    def __init__(self, **kwargs):
        super(Gray2RGB, self).__init__(trainable=False, **kwargs)

    def call(self, inputs, **kwargs):
        return tf.image.grayscale_to_rgb(inputs)


custom_layers['Gray2RGB'] = Gray2RGB


class CastUInt2Float(Layer):
    """Cast input type from uint8 to float32

    """

    def __init__(self, **kwargs):
        super(CastUInt2Float, self).__init__(dtype='uint8', **kwargs)

    def call(self, inputs, **kwargs):
        return tf.cast(inputs, dtype=tf.float32)


custom_layers['CastUInt2Float'] = CastUInt2Float


class VGGNormalize(Layer):
    '''
    Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG network.
    '''

    def __init__(self, **kwargs):
        super(VGGNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        # No exact substitute for set_subtensor in tensorflow
        # So we subtract an approximate value
        if x.shape[-1] == 1:
            # 'Gray'->'RGB'
            x = tf.image.grayscale_to_rgb(x)
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        x -= 114
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


custom_layers['VGGNormalize'] = VGGNormalize


class RecursiveConv2D(Layer):
    """Recursive convolution layer used in DRCN

    """

    def __init__(self, units, **kwargs):
        super(RecursiveConv2D, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.shared_kernel = self.add_weight('RecursiveKernel', [3, 3, 256, 256], initializer='he_normal')
        super(RecursiveConv2D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outp = [K.relu(K.conv2d(inputs, self.shared_kernel, padding='same'))]
        for i in range(1, self.units):
            outp.append(K.relu(K.conv2d(outp[i - 1], self.shared_kernel, padding='same')))
        return [outp]

    def compute_output_shape(self, input_shape):
        return [input_shape] * self.units

    def compute_mask(self, inputs, mask=None):
        return [mask] * self.units


class WeightedAdd(Layer):
    """Merge layers by weights

    """

    def __init__(self, size, **kwargs):
        super(WeightedAdd, self).__init__(**kwargs)
        self.size = size

    def build(self, input_shape):
        self.added_weights = self.add_weight('AddedWeights', [len(input_shape)], initializer='uniform')
        super(WeightedAdd, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs[0]
