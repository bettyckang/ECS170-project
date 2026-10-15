'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import pandas as pd


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    # stage 2 data comes pre-partitioned, so we load the train and test files separately
    dataset_train_file_name = None
    dataset_test_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_file(self, file_name):
        # each line: label,feature1,...,feature784 (no header row)
        df = pd.read_csv(self.dataset_source_folder_path + file_name, header=None)
        X = df.iloc[:, 1:].to_numpy(dtype='float32')
        y = df.iloc[:, 0].to_numpy(dtype='int64')
        return {'X': X, 'y': y}

    def load(self):
        print('loading data...')
        train = self.load_file(self.dataset_train_file_name)
        test = self.load_file(self.dataset_test_file_name)
        print('train:', train['X'].shape, 'test:', test['X'].shape)
        return {'train': train, 'test': test}
