import argparse, json

from functools import partial
from pathlib import Path
from keras import optimizers
from toolbox.data import load_set
from toolbox.models import get_model
from toolbox.experiment import Experiment

parser = argparse.ArgumentParser()
parser.add_argument('param_file', type=Path)
args = parser.parse_args()
param = json.load(args.param_file.open())

# Model
scale = param['scale']
channel = param['channel'] if 'channel' in param else 1
random = param['random'] if 'random' in param else 0
pre_upsample = param['pre_upsample'] if 'pre_upsample' in param else False
build_model = partial(get_model(param['model']['name']),
                      **param['model']['params'])
if 'optimizer' in param:
    optimizer = getattr(optimizers, param['optimizer']['name'].lower())
    optimizer = optimizer(**param['optimizer']['params'])
else:
    optimizer = 'adam'

# Data
load_set = partial(load_set,
                   lr_sub_size=param['lr_sub_size'],
                   lr_sub_stride=param['lr_sub_stride'],
                   pre_upsample=pre_upsample,
                   random=random)

# Training
expt = Experiment(scale=param['scale'], channel=channel,
                  load_set=load_set, build_model=build_model,
                  optimizer=optimizer,
                  save_dir=Path('./results') / param['save_dir'])
expt.train(train_set=param['train_set'], val_set=param['val_set'],
           epochs=param['epochs'], resume=True)

# Evaluation
for test_set in param['test_sets']:
    expt.test(test_set=test_set, pre_upsample=pre_upsample)

# Export tensorflow .pb model
expt.export_pb_model(['input_lr'], ['output_hr'],
                     Path('./results') / param['save_dir'] / 'model.pb')
