import time, matplotlib

matplotlib.use('Agg')
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from functools import partial
from pathlib import Path

from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import img_to_array
from keras.utils import plot_model
# export pb model
import tensorflow as tf

from toolbox.data import load_image_pair, load_set
from toolbox.image import array_to_img
from toolbox.metrics import psnr
from toolbox.dataset import DATASET
from toolbox.loss import get_loss
from toolbox.models import *
from toolbox.callbacks import LearningRateScheduler, SchedulerByLossGrad


class Experiment(object):
    def __init__(self,
                 scale=3,
                 channel=1,
                 build_model=None,
                 optimizer='adam',
                 loss={'name': 'mse'},
                 lr_sub_size=11,
                 lr_sub_stride=5,
                 random=0,
                 save_dir='.',
                 **kwargs):
        self.scale = scale
        self.channel = channel
        self.load_set = partial(load_set,
                                scale=scale,
                                lr_sub_size=lr_sub_size,
                                lr_sub_stride=lr_sub_stride,
                                random=random)
        self.build_model = partial(build_model, scale=scale)
        self.optimizer = optimizer
        self.loss = get_loss(**loss)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.save_dir / 'config.yaml'
        self.model_file = self.save_dir / 'model.hdf5'

        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history_file = self.train_dir / 'history.csv'
        self.weights_dir = self.train_dir / 'weights'
        self.weights_dir.mkdir(exist_ok=True)

        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)

        self.model = None

    def weights_file(self, epoch=None):
        if epoch is None:
            return self.weights_dir / 'ep{epoch:04d}.hdf5'
        else:
            return self.weights_dir / f'ep{epoch:04d}.hdf5'

    @property
    def latest_epoch(self):
        try:
            return pd.read_csv(str(self.history_file))['epoch'].iloc[-1]
        except (FileNotFoundError, pd.io.common.EmptyDataError):
            pass
        return -1

    @staticmethod
    def _ensure_dimension(array, dim):
        while len(array.shape) < dim:
            array = array[np.newaxis, ...]
        return array

    @staticmethod
    def _ensure_channel(array, c):
        array_c = array.shape[-1]
        if array_c < c:
            array = np.concatenate((array, np.ones(list(array.shape[0:-1]) + [c - array_c])), axis=-1)
        return array[..., 0:c]

    def pre_process(self, array):
        array = self._ensure_dimension(array, 4)
        array = self._ensure_channel(array, self.channel)
        return array

    def post_process(self, array, auxiliary_array=None):
        if self.channel == 1:
            array = np.concatenate([array, auxiliary_array[..., 1:]], axis=-1)
        array = array[..., 0:3]
        array = np.clip(array, 0, 255)
        return array

    def inverse_post_process(self, array):
        array = self._ensure_dimension(array, 4)
        array = self._ensure_channel(array, self.channel)
        return array

    def compile(self, model):
        """Compile model with default settings."""
        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=[psnr])
        return model

    def train(self, train_set='91-image', val_set='Set5', batch_size=128, epochs=1, resume=True):
        # Compile model
        self.model = self.build_model(self.channel)
        model = self.model
        model.summary()
        # Inherit weights
        skip_training = False
        if resume:
            latest_epoch = self.latest_epoch
            if latest_epoch > -1:
                weights_file = self.weights_file(epoch=latest_epoch + 1)
                model.load_weights(str(weights_file))
            initial_epoch = latest_epoch + 1
            if initial_epoch >= epochs:
                skip_training = True
        else:
            initial_epoch = 0

        # Save model architecture
        # Currently in Keras 2 it's not possible to load a model with custom
        # layers. So we just save it without checking consistency.
        self.config_file.write_text(model.to_yaml())
        # Set up callbacks
        callbacks = []
        callbacks += [ModelCheckpoint(str(self.model_file))]
        callbacks += [ModelCheckpoint(str(self.weights_file()),
                                      save_weights_only=True)]
        callbacks += [CSVLogger(str(self.history_file), append=resume)]
        # callbacks += [LearningRateScheduler(SchedulerByLossGrad(0.02,2))]

        # Load and process data
        if not skip_training:
            x_train, y_train = self.load_set(train_set, mode='train')
            x_val, y_val = self.load_set(val_set, mode='val')
            x_train, x_val = [self.pre_process(x)
                              for x in [x_train, x_val]]
            y_train, y_val = [self.inverse_post_process(y)
                              for y in [y_train, y_val]]
            # Train
            # model.input.set_shape([None] + list(x_train.shape[1:]))
            # model.output.set_shape([None] + list(y_train.shape[1:]))
            model = self.compile(model)
            model.fit(x_train, y_train, batch_size=batch_size,
                      epochs=epochs, callbacks=callbacks,
                      validation_data=(x_val, y_val), initial_epoch=initial_epoch)

        # Plot metrics history
        prefix = str(self.history_file).rsplit('.', maxsplit=1)[0]
        df = pd.read_csv(str(self.history_file))
        epoch = df['epoch']
        for metric in ['Loss', 'PSNR']:
            train = df[metric.lower()]
            val = df['val_' + metric.lower()]
            plt.figure()
            plt.plot(epoch, train, label='train')
            plt.plot(epoch, val, label='val')
            plt.legend(loc='best')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.savefig('.'.join([prefix, metric.lower(), 'png']))
            plt.close()

    def test(self, test_set='Set5', metrics=tuple([psnr])):
        print('Test on', test_set)
        image_dir = self.test_dir / test_set
        image_dir.mkdir(exist_ok=True)

        # Evaluate metrics on each image
        model = self.compile(self.build_model(self.channel)) if not self.model else self.model
        rows = []
        for image_path in DATASET[test_set.upper()].val:
            rows += [self.test_on_image(model,
                                        image_path,
                                        str(image_dir / Path(image_path).stem),
                                        metrics=metrics)]
        df = pd.DataFrame(rows)

        # Compute average metrics
        row = pd.Series()
        row['name'] = 'average'
        for col in df:
            if col != 'name':
                row[col] = df[col].mean()
        df = df.append(row, ignore_index=True)

        df.to_csv(str(self.test_dir / f'{test_set}/metrics.csv'))

    def test_file(self, filepaths, metrics=tuple([psnr])):
        print('Test on individual files')
        image_dir = self.test_dir
        # Evaluate metrics on each image
        model = self.compile(self.build_model(self.channel)) if not self.model else self.model
        rows = []
        for image_path in filepaths:
            rows += [self.test_on_image(model,
                                        image_path,
                                        str(image_dir / Path(image_path).stem),
                                        metrics=metrics)]

    def test_on_image(self, model, path, prefix, suffix='jpg', metrics=tuple([psnr])):
        # Load images
        lr_image, hr_image = load_image_pair(path, scale=self.scale, mode='YCbCr')

        # Generate bicubic image
        x = img_to_array(lr_image)[np.newaxis, ...]
        bicubic_model = bicubic(3, scale=self.scale)
        y = bicubic_model.predict_on_batch(x)
        cu_image = array_to_img(np.clip(y[0], 0, 255))

        # Generate output image and measure run time
        x = self.pre_process(x).astype('uint8')
        if self.model_file.exists():
            model.load_weights(str(self.model_file))
        start = time.perf_counter()
        y_pred = model.predict_on_batch(x)
        end = time.perf_counter()
        output_array = self.post_process(y_pred[0], y[0])
        output_image = array_to_img(output_array, mode='YCbCr')

        # Record metrics
        row = pd.Series()
        row['name'] = Path(path).stem
        row['time'] = end - start
        y_true = self.inverse_post_process(img_to_array(hr_image))
        for metric in metrics:
            row[metric.__name__] = K.eval(metric(y_true[..., :3], y_pred[..., :3]))

        # Save images
        images_to_save = []
        images_to_save += [(hr_image, 'original')]
        images_to_save += [(output_image, 'output')]
        # images_to_save += [(lr_image, 'input')]
        images_to_save += [(cu_image, 'bicubic')]
        for img, label in images_to_save:
            img.convert(mode='RGB').save('.'.join([prefix, label, suffix]), quality=95)

        return row

    def export_pb_model(self, input_name=None, output_name=None,
                        pb_model_path='model.pb', bgr=False):
        if K.backend() != 'tensorflow':
            print("Can't export model for keras backend is %s" % K.backend())
            return
        model = self.compile(self.build_model(self.channel)) if not self.model else self.model
        if self.model_file.exists():
            model.load_weights(str(self.model_file))
            model = composeModel(4, self.scale, model, bgr)
            model = self.compile(model)
            model.summary()
        try:
            plot_model(model, str(self.save_dir / 'model.png'))
        except ImportError:
            print('Fail to plot model, can not find dot.exe. Do you install graphviz?')
        sess = K.get_session()
        if not output_name:
            output_name = [n for n in model.output_names]
        casted_outputs = []
        for outp in model.outputs:
            casted_outputs.append(K.cast(K.clip(outp, 0, 255), 'uint8'))
        for outp, name in zip(casted_outputs, output_name):
            tf.identity(outp, name=name)
        for inp, name in zip(model.inputs, input_name):
            tf.identity(inp, name=name)
        const_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), output_name)
        write_dir = Path(pb_model_path).parent
        write_name = Path(pb_model_path).stem + '.pb'
        tf.train.write_graph(const_graph, str(write_dir), write_name, False)
