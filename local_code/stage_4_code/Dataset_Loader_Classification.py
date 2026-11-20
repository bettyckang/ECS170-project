'''
Concrete IO class for the stage 4 text classification dataset (IMDb movie reviews)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import numpy as np
from local_code.base_class.dataset import dataset
from local_code.stage_4_code.Text_Utils import clean_review, build_vocabulary, load_glove, PAD, UNK


class Dataset_Loader_Classification(dataset):
    data = None
    # folder holding train/{pos,neg} and test/{pos,neg}
    dataset_source_folder_path = None
    glove_file_path = None
    embedding_dim = 100
    # words seen fewer than min_count times in the training reviews become <unk>
    min_count = 5
    # longer reviews keep their LAST max_len words: reviews usually end with the verdict,
    # and a recurrent network remembers recent words best
    max_len = 256
    # 'last' keeps the end of long reviews, 'first' keeps the beginning (an ablation)
    keep = 'last'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def read_split(self, split):
        texts, labels = [], []
        for label, folder in [(1, 'pos'), (0, 'neg')]:
            path = os.path.join(self.dataset_source_folder_path, split, folder)
            for file_name in sorted(os.listdir(path)):
                if file_name.endswith('.txt'):
                    with open(os.path.join(path, file_name), encoding='utf-8') as f:
                        texts.append(clean_review(f.read()))
                    labels.append(label)
        return texts, np.array(labels, dtype='int64')

    def encode(self, token_lists, index):
        # word ids padded with 0 (<pad>) at the end; lengths say where each review really ends
        X = np.zeros((len(token_lists), self.max_len), dtype='int64')
        lengths = np.zeros(len(token_lists), dtype='int64')
        unk = index[UNK]
        for i, tokens in enumerate(token_lists):
            ids = [index.get(w, unk) for w in tokens]
            ids = (ids[-self.max_len:] if self.keep == 'last' else ids[:self.max_len]) or [unk]
            X[i, :len(ids)] = ids
            lengths[i] = len(ids)
        return X, lengths

    def load(self):
        print('loading data...')
        train_tokens, train_y = self.read_split('train')
        test_tokens, test_y = self.read_split('test')
        # the vocabulary comes from the training reviews only, so nothing about the test set leaks in
        vocab = build_vocabulary(train_tokens, self.min_count, specials=(PAD, UNK))
        index = {w: i for i, w in enumerate(vocab)}
        train_X, train_len = self.encode(train_tokens, index)
        test_X, test_len = self.encode(test_tokens, index)
        embedding, coverage = load_glove(self.glove_file_path, vocab, self.embedding_dim)
        print('train:', train_X.shape, 'test:', test_X.shape, 'vocabulary: %d words (%.1f%% found in GloVe)'
              % (len(vocab), 100 * coverage))
        return {'train': {'X': train_X, 'lengths': train_len, 'y': train_y},
                'test': {'X': test_X, 'lengths': test_len, 'y': test_y},
                'vocab': vocab, 'embedding': embedding,
                'raw_lengths': {'train': np.array([len(t) for t in train_tokens]),
                                'test': np.array([len(t) for t in test_tokens])}}
