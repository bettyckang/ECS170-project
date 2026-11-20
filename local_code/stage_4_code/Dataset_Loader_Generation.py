'''
Concrete IO class for the stage 4 text generation dataset (short jokes)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import csv
import numpy as np
from local_code.base_class.dataset import dataset
from local_code.stage_4_code.Text_Utils import tokenize_joke, build_vocabulary, load_glove, PAD, UNK, EOS


class Dataset_Loader_Generation(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = 'data'
    glove_file_path = None
    embedding_dim = 100

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, encoding='utf-8') as f:
            jokes = [row['Joke'] for row in csv.DictReader(f)]
        token_lists = [tokenize_joke(joke) for joke in jokes]
        # every word is kept (no minimum count): the model should be able to reproduce any joke
        vocab = build_vocabulary(token_lists, min_count=1, specials=(PAD, UNK, EOS))
        index = {w: i for i, w in enumerate(vocab)}
        # each joke ends with <eos>, so the model learns where a joke stops
        sequences = [[index[w] for w in tokens] + [index[EOS]] for tokens in token_lists]
        max_len = max(len(s) for s in sequences)
        X = np.zeros((len(sequences), max_len), dtype='int64')
        lengths = np.array([len(s) for s in sequences], dtype='int64')
        for i, s in enumerate(sequences):
            X[i, :len(s)] = s
        embedding, coverage = load_glove(self.glove_file_path, vocab, self.embedding_dim)
        print('jokes:', len(jokes), 'longest: %d tokens' % max_len,
              'vocabulary: %d tokens (%.1f%% found in GloVe)' % (len(vocab), 100 * coverage))
        # there is no test set: the task is to generate jokes and compare them with the training data
        return {'train': {'X': X, 'lengths': lengths, 'tokens': token_lists, 'text': jokes},
                'vocab': vocab, 'embedding': embedding}
