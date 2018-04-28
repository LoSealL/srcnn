from toolbox.models import *
from toolbox.dataset import *
from toolbox.data import *
from toolbox.image import *
from toolbox.layers import *
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.utils import plot_model
from keras.losses import get

def mse_calc():
    from keras.applications.vgg16 import VGG16
    from keras.models import Model

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
    from tensorflow.python.platform import gfile
    with tf.Session() as sess:
        model_filename = '../results/dcscn-mse-sc3/model.pb'
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
        g = sess.graph
        url = DATASET['SET5'].val[0]
        img, _ = load_image_pair('../results/help.png')
        bicubic_rescale(img, 3, 'RGB').save('../results/test_cmp.png')
        img = img_to_array(img)
        img = np.concatenate((img, np.ones(list(img.shape[0:-1]) + [1])), axis=-1)
        img = np.expand_dims(img, axis=0)
        out = sess.run("import/output_hr:0", feed_dict={"import/input_lr_rgb:0": img})
        img = array_to_img(out[0][..., :3], 'RGB')
        img.save('../results/test.png')


def test_data_yuv():
    _, y = load_video_set_raw('MCL-V', 3, [1080, 1920], stride=480, seq_depth=5)
    # [N, S, H, W, C]
    shape = y.shape
    for i in range(shape[0]):
        path = Path('test') / ('s' + str(i))
        path.mkdir(parents=True, exist_ok=True)
        for j in range(shape[1]):
            img_path = path / f'img_{j:03d}.png'
            array_to_img(y[i, j]).convert('RGB').save(str(img_path))


def train_spmc():
    model = spmc()
    model.compile('adam', 'mse')
    model.summary()
    plot_model(model, 'test_model.png')
    x, y = load_video_set_raw('MCL-V', 3, [1080, 1920], stride=480, seq_depth=2, method='train')
    print("video loaded " + str(x.shape[0]) + " batches")
    x_gray = x[..., :1]
    label = x[:, 0, ..., :1]
    model.fit(x_gray, label, batch_size=100, epochs=100)
    y = model.predict_on_batch(x[-2:-1, ...])
    array_to_img(y[0], mode='L').convert('RGB').save('test.png')


if __name__ == '__main__':
    train_spmc()
