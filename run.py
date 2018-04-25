import argparse, json

from functools import partial
from pathlib import Path
from keras import optimizers

from toolbox.models import get_model
from toolbox.experiment import Experiment

parser = argparse.ArgumentParser()
parser.add_argument('param_file', type=Path)
parser.add_argument('--export-only', type=bool, default=False)
args = parser.parse_args()
param = json.load(args.param_file.open())
save_dir = Path(args.param_file).stem

# Model
scale = param['scale']
channel = param['channel'] if 'channel' in param else 1
random = param['random'] if 'random' in param else 0
output_format = param['outputformat'] if 'outputformat' in param else 'RGB'
build_model = partial(get_model(param['model']['name']),
                      **param['model']['params'])
if 'optimizer' in param:
    optimizer = getattr(optimizers, param['optimizer']['name'].lower())
    optimizer = optimizer(**param['optimizer']['params'])
else:
    optimizer = 'adam'
if 'loss' in param:
    loss = param['loss']
else:
    loss = {'name': 'mse'}
if output_format.upper() == 'RGB':
    export_bgr = False
else:
    export_bgr = True

expt = Experiment(scale=param['scale'], channel=channel,
                  build_model=build_model,
                  optimizer=optimizer, loss=loss,
                  lr_sub_size=param['lr_sub_size'],
                  lr_sub_stride=param['lr_sub_stride'],
                  random=random,
                  save_dir=Path('./results') / save_dir)

if args.export_only:
    expt.export_pb_model(['input_lr'], ['output_hr'],
                         Path('./results') / save_dir / 'model.pb', export_bgr)
    exit()

# Training
expt.train(train_set=param['train_set'], val_set=param['val_set'],
           epochs=param['epochs'], batch_size=param['batch'], resume=True)

# Evaluation
if 'test_sets' in param:
    for test_set in param['test_sets']:
        expt.test(test_set=test_set)
if 'test_files' in param:
    expt.test_file(param['test_files'])

# Export tensorflow .pb model
expt.export_pb_model(['input_lr'], ['output_hr'],
                     Path('./results') / save_dir / 'model.pb', export_bgr)
