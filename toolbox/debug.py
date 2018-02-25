from toolbox.models import *
from toolbox.dataset import *
from toolbox.data import *
from toolbox.image import *
from toolbox.layers import *
import keras.backend as K
import keras


def model_as_layer():
    img_url = DATASET['SET5'].test
    img = [load_image_pair(i) for i in img_url]

    lr, hr, cu = [], [], []
    lr.extend((img_to_array(i[0]) for i in img))
    hr.extend((img_to_array(i[1]) for i in img))
    cu.extend((img_to_array(i[2]) for i in img))

    net = vgg16(input_shape=hr[0].shape)
    net.compile('adam', 'mse')

    pred = net.predict(hr[0][None, ...])[0]

    index = 0
    for i in range(pred.shape[-1]):
        f = pred[..., i]
        array_to_img(f, None).save(f'../results/debug/vg1{index:03d}.jpg')
        index += 1

    inp = keras.layers.Input(hr[0].shape)
    x = VGGNormalize()(inp)
    outp = net(x)
    model = keras.models.Model(inp, outp)
    model.compile('adam', 'mse')

    pred = model.predict(hr[0][None, ...])[0]
    index = 0
    for i in range(pred.shape[-1]):
        f = pred[..., i]
        array_to_img(f, None).save(f'../results/debug/vg2{index:03d}.jpg')
        index += 1


def compare_my_vgg_vs_VGG_keras():
    from keras.applications.vgg16 import VGG16
    from keras.models import Model
    import numpy as np

    net1 = vgg16(input_shape=[None, None, 3], include_top=False)
    net2 = VGG16(input_shape=[None, None, 3], include_top=False)

    layer_name = 'block2_conv2'
    L1 = net1.get_layer(layer_name)
    L2 = net2.get_layer(layer_name)

    model1 = Model(net1.input, L1.output)
    model2 = Model(net2.input, L2.output)

    x = np.random.randint(0, 255, [256, 256, 3])
    x = K.variable(x[None, ...])
    x = VGGNormalize()(x)

    y1 = model1(x)
    y2 = model2(x)

    y1 = K.eval(y1)
    y2 = K.eval(y2)

    cmp = y1 == y2
    print("Result: " + str(cmp.all()))


def mse_calc():
    from keras.applications.vgg16 import VGG16
    from keras.models import Model
    import numpy as np

    img_url = DATASET['SET5'].test
    img = [load_image_pair(i) for i in img_url]

    lr, hr, cu = [], [], []
    lr.extend((img_to_array(i[0]) for i in img))
    hr.extend((img_to_array(i[1]) for i in img))
    cu.extend((img_to_array(i[2]) for i in img))
    net2 = VGG16(input_shape=[None, None, 3], include_top=False)

    layer_name = 'block2_conv2'
    L2 = net2.get_layer(layer_name)

    model2 = Model(net2.input, L2.output)

    x = hr[0]
    x = K.variable(x[None, ...])
    x = VGGNormalize()(x)

    y2 = model2(x)
    print(y2.shape)
    print(K.eval(K.max(y2)))
    s1 = K.mean(K.square(y2))
    print(K.eval(s1))
    s2 = K.mean(s1)
    print(K.eval(s2))


def test_pb_model():
    import tensorflow as tf
    from tensorflow.python.platform import gfile
    with tf.Session() as sess:
        model_filename = '../results/srcnn-example/model.pb'
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
        g = sess.graph
        img = np.ones([1, 256, 256, 1])
        a = sess.graph.get_tensor_by_name("import/input_4:0")
        b = sess.graph.get_tensor_by_name("import/CastedClipOutput:0")
        sess.run("import/CastedClipOutput:0", feed_dict={"import/input_4:0": img})


if __name__ == '__main__':
    test_pb_model()
