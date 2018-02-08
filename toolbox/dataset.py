"""
Copyright 2017 Intel.Corp
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Date: 2018-1-11

offline dataset framework
"""
import json
from pathlib import Path
from toolbox.paths import repo_dir

DATASET = dict()


class Dataset(object):
    """globs all images in the data path"""

    def __init__(self, train, eval, test, train_filter, eval_filter, test_filter):
        """Specify data path

        Args:
           :param train: training data
           :param eval: evaluation data
           :param test: testing data
           :param train_filter, eval_filter, test_filter: filename filters
        """
        self._train = self._preprocess(train, train_filter)
        self._val = self._preprocess(eval, eval_filter)
        self._test = self._preprocess(test, test_filter)

    @staticmethod
    def _preprocess(url, filters):
        if not url or not Path(url).exists(): return
        if not isinstance(filters, list):
            raise TypeError("Filters is not in list!")
        files = []
        for _f in filters:
            files += [str(x) for x in Path(url).glob(_f)]
        return files

    def __getattr__(self, item):
        if item == 'train_size':
            return len(self._train)
        elif item == 'val_size':
            return len(self._val)
        elif item == 'test_size':
            return len(self._test)
        elif item == 'train':
            if not self._train:
                raise ValueError('Untrainable')
            else:
                return self._train
        elif item == 'val':
            if not self._val:
                raise ValueError('Unvalidatable')
            else:
                return self._val
        elif item == 'test':
            if not self._test:
                raise ValueError('Untestable')
            else:
                return self._test


with open(repo_dir / 'config/datasets.json', 'r') as fd:
    config = json.load(fd)
    for k, v in config.items():
        _train, _train_filter = None, []
        _eval, _eval_filter = None, []
        _test, _test_filter = None, []
        if 'train' in v:
            _train = v['train']['url']
            _train_filter = v['train']['filter'].split(',')
        if 'eval' in v:
            _eval = v['eval']['url']
            _eval_filter = v['eval']['filter'].split(',')
        if 'test' in v:
            _test = v['test']['url']
            _test_filter = v['test']['filter'].split(',')
        DATASET[k] = Dataset(
            _train, _eval, _test,
            train_filter=_train_filter,
            eval_filter=_eval_filter,
            test_filter=_test_filter)
