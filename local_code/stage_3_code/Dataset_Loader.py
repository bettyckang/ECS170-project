'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import pickle
import numpy as np


class SafeUnpickler(pickle.Unpickler):
    # the stage 3 files are pickles; loading a pickle can run arbitrary code,
    # so we only allow the numpy array types these files actually contain
    ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'dtype'), ('numpy', 'ndarray')}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError('blocked unexpected object in dataset file: ' + module + '.' + name)
        return super().find_class(module, name)


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    # one of 'MNIST', 'ORL', 'CIFAR' (also the file name inside data/stage_3_data/)
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def to_arrays(self, instances):
        X = np.array([instance['image'] for instance in instances], dtype='float32') / 255.0
        y = np.array([instance['label'] for instance in instances], dtype='int64')
        if self.dataset_source_file_name == 'MNIST':
            # (N, 28, 28) gray-scale -> (N, 1, 28, 28)
            X = X[:, np.newaxis, :, :]
        elif self.dataset_source_file_name == 'ORL':
            # the R, G and B channels are identical gray-scale copies, so we keep one: (N, 112, 92, 3) -> (N, 1, 112, 92)
            X = X[:, np.newaxis, :, :, 0]
            # labels are 1-40; PyTorch's loss functions expect classes numbered from 0
            y = y - 1
        elif self.dataset_source_file_name == 'CIFAR':
            # PyTorch convolutions expect channels first: (N, 32, 32, 3) -> (N, 3, 32, 32)
            X = X.transpose(0, 3, 1, 2)
        return {'X': np.ascontiguousarray(X), 'y': y}

    def load(self):
        print('loading data...')
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            raw = SafeUnpickler(f).load()
        train = self.to_arrays(raw['train'])
        test = self.to_arrays(raw['test'])
        print('train:', train['X'].shape, 'test:', test['X'].shape)
        return {'train': train, 'test': test}
