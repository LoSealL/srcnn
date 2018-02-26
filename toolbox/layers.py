from keras.engine.topology import Layer, InputSpec
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
        return K.cast(inputs, dtype=K.floatx())


custom_layers['CastUInt2Float'] = CastUInt2Float


class VGGNormalize(Layer):
    """Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG network.

    """

    def __init__(self, **kwargs):
        super(VGGNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(VGGNormalize, self).build(input_shape)

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


class DRRNResidualBlock(Layer):
    """

    """

    def __init__(self, units, filters=128, kernel_size=3, **kwargs):
        self.U = units
        self.f = filters
        self.k = kernel_size
        self.w = []
        self.b = []
        self.axis = -1
        self.epsilon = 1e-3
        super(DRRNResidualBlock, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        self.w.append(self.add_weight(f'InitConv',
                                      shape=[self.k, self.k, self.f, self.f],
                                      initializer='he_normal'))
        self.b.append(self.add_weight(f'InitBias',
                                      shape=(self.f,),
                                      initializer='zeros'))
        for i in range(2):
            self.w.append(self.add_weight(f'ResidualConv{i}',
                                          shape=[self.k, self.k, self.f, self.f],
                                          initializer='he_normal'))
            self.b.append(self.add_weight(f'ResidualBias{i}',
                                          shape=(self.f,),
                                          initializer='zeros'))
        shape = (input_shape[self.axis],)
        self.gamma = self.add_weight(shape=shape,
                                     name='bn_gamma',
                                     initializer='ones')
        self.beta = self.add_weight(shape=shape,
                                    name='bn_beta',
                                    initializer='zeros')
        self.moving_mean = self.add_weight(
            shape=shape,
            name='bn_moving_mean',
            initializer='zeros',
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='bn_moving_variance',
            initializer='ones',
            trainable=False)
        self.momentum = 0.99
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        inp = K.conv2d(inputs, self.w[0], padding='same')
        inp = K.bias_add(inp, self.b[0])
        recursive_inp = inp
        for i in range(1, self.U + 1):
            x = self._bn(recursive_inp, training)
            x = K.relu(x)
            x = K.conv2d(x, self.w[1], padding='same')
            x = K.bias_add(x, self.b[1])
            x = self._bn(x, training)
            x = K.relu(x)
            x = K.conv2d(x, self.w[2], padding='same')
            x = K.bias_add(x, self.b[2])
            recursive_inp = x + inp
        recursive_outp = recursive_inp + inp
        return recursive_outp

    def compute_output_shape(self, input_shape):
        return input_shape

    def _bn(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                broadcast_beta = K.reshape(self.beta, broadcast_shape)
                broadcast_gamma = K.reshape(self.gamma,
                                            broadcast_shape)
                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)
