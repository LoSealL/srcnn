from keras.callbacks import Callback
import keras.backend as K
import numpy as np


class LearningRateScheduler(Callback):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index and loss as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.diff_loss = 0

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        loss = logs.get('val_loss')
        lr = self.schedule(epoch, loss)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (epoch + 1, lr))


class SchedulerByLossGrad(object):

    def __init__(self, init_lr, decay, window=5):
        self.initial_rate = init_lr
        self.current_rate = init_lr
        self.decay = decay
        self.prev_loss = 0
        self.window = window
        self.loss_gradient = [] * window
        self.update_counter = 0

    def __call__(self, epoch, loss):
        if len(self.loss_gradient) >= self.window:
            self.loss_gradient.pop(0)
        self.loss_gradient.append(loss - self.prev_loss)
        self.prev_loss = loss
        if all([grad <= 0 for grad in self.loss_gradient]):
            lr = self.current_rate / (K.epsilon() + self.decay)
            self.current_rate = lr
        else:
            lr = self.current_rate
        return lr
