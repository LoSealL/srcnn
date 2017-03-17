from pathlib import Path

from keras.models import model_from_yaml
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd

from toolbox.data import load_image_pair
from toolbox.metrics import psnr
from toolbox.metrics import tf_eval
from toolbox.paths import data_dir
from toolbox.preprocessing import array_to_img
from toolbox.preprocessing import bicubic_resize
from toolbox.preprocessing import identity
from toolbox.visualization import plot_history


class Experiment(object):
    def __init__(self, scale=3, model=None, preprocess=identity, load_set=None,
                 save_dir='.'):
        self.scale = scale
        self.model = model
        self.preprocess = preprocess
        self.load_set = load_set
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.config_file = self.train_dir / 'config.yaml'
        self.history_file = self.train_dir / 'history.csv'
        self.model_file = self.train_dir / 'model.hdf5'
        self.weights_dir = self.train_dir / 'weights'
        self.weights_dir.mkdir(exist_ok=True)

        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)

    def weights_file(self, epoch=None):
        if epoch is None:
            return self.weights_dir / 'ep{epoch:04d}.hdf5'
        else:
            return self.weights_dir / f'ep{epoch:04d}.hdf5'

    @property
    def latest_epoch(self):
        try:
            return pd.read_csv(str(self.history_file))['epoch'].iloc[-1]
        except FileNotFoundError or pd.io.common.EmptyDataError:
            pass
        return -1

    def train(self, train_set='91-image', val_set='Set5', epochs=1,
              resume=True):
        # Check architecture
        if resume and self.config_file.exists():
            # Check architecture consistency
            saved_model = model_from_yaml(self.config_file.read_text())
            if self.model.get_config() != saved_model.get_config():
                raise ValueError('Model architecture has changed.')
        else:
            # Save architecture
            self.config_file.write_text(self.model.to_yaml())

        # Set up callbacks
        callbacks = []
        callbacks += [ModelCheckpoint(str(self.model_file))]
        callbacks += [ModelCheckpoint(str(self.weights_file()),
                                      save_weights_only=True)]
        callbacks += [CSVLogger(str(self.history_file), append=resume)]

        # Inherit weights
        if resume:
            latest_epoch = self.latest_epoch
            if latest_epoch > -1:
                weights_file = self.weights_file(epoch=latest_epoch)
                self.model.load_weights(str(weights_file))
            initial_epoch = latest_epoch + 1
        else:
            initial_epoch = 0

        # Load data and train
        x_train, y_train = self.load_set(train_set)
        x_val, y_val = self.load_set(val_set)
        self.model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks,
                       validation_data=(x_val, y_val),
                       initial_epoch=initial_epoch)

        # Make diagnostic plots
        plot_history(str(self.history_file))

    def test(self, test_set='Set5', metrics=[psnr]):
        print('Test on', test_set)
        output_dir = self.test_dir / test_set
        output_dir.mkdir(exist_ok=True)
        rows = []
        for image_path in (data_dir / test_set).glob('*'):
            rows += [self.test_on_image(str(image_path),
                                        str(output_dir / image_path.stem))]
        df = pd.DataFrame(rows)
        row = pd.Series()
        row['name'] = 'average'
        row['psnr'] = df['psnr'].mean()
        df = df.append(row, ignore_index=True)
        df.to_csv(str(self.test_dir / f'metrics_{test_set}.csv'))

        # Tabulate metrics history
        model = model_from_yaml(self.config_file.read_text())
        x, y_true = self.load_set(test_set)
        epochs = self.latest_epoch + 1
        rows = []
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            row = pd.Series()
            row['epoch'] = epoch
            model.load_weights(str(self.weights_file(epoch)))
            y_pred = model.predict(x)
            for metric in metrics:
                row[metric.__name__] = tf_eval(metric(y_true, y_pred))
                print(metric.__name__, row[metric.__name__])
            rows += [row]
        df = pd.DataFrame(rows)
        df.to_csv(str(self.test_dir / f'metrics_{test_set}_history.csv'))

    def test_on_image(self, path, prefix, suffix='png'):
        lr_image, hr_image = load_image_pair(path, scale=self.scale)

        model = self.model
        x = img_to_array(self.preprocess(lr_image))
        x = x[np.newaxis, :, :, 0:1]
        y_pred = model.predict_on_batch(x)

        row = pd.Series()
        row['name'] = Path(path).stem
        y_true = img_to_array(hr_image)[np.newaxis, ...]
        row['psnr'] = tf_eval(psnr(y_true, y_pred))

        bicubic_image = bicubic_resize(lr_image, self.scale)
        output_array = img_to_array(bicubic_image)
        output_array[:, :, 0] = y_pred[0, :, :, 0]
        output_image = array_to_img(output_array, mode='YCbCr')

        images_to_save = []
        images_to_save += [(hr_image, 'original')]
        images_to_save += [(bicubic_image, 'bicubic')]
        images_to_save += [(output_image, 'output')]
        images_to_save += [(lr_image, 'input')]
        for img, label in images_to_save:
            img.convert(mode='RGB').save('.'.join([prefix, label, suffix]))

        return row