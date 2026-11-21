'''
Concrete Evaluate class for the joke generator: compares generated jokes with the training jokes
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from collections import defaultdict
from difflib import SequenceMatcher
import numpy as np
from local_code.base_class.evaluate import evaluate
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics


class Evaluate_Generation(evaluate):
    data = None
    start_words = 3

    def evaluate(self):
        print('evaluating performance...')
        # next-word accuracy, precision, recall and F1 (the word after each prefix, as a classification
        # over the whole vocabulary); weighted by how often each word appears
        word_metrics = Evaluate_Metrics('next word', '')
        word_metrics.averages = ('weighted', 'macro')
        word_metrics.data = {'true_y': self.data['true_y'], 'pred_y': self.data['pred_y']}
        scores = {'next_word_' + k: v for k, v in word_metrics.evaluate().items()}

        # several jokes can share a start ("what do you" begins 192 of them), so a generated joke
        # is compared with every training joke that has the same three start words
        by_start = defaultdict(list)
        for tokens in self.data['jokes']:
            by_start[tuple(tokens[:self.start_words])].append(tokens)
        exact, similarity = [], []
        for start, tokens in self.data['generated'].items():
            references = by_start[start]
            exact.append(any(tokens == ref for ref in references))
            # word-level similarity in [0, 1]: 2 x (words matched in order) / (total words in both jokes)
            similarity.append(max(SequenceMatcher(None, tokens, ref).ratio() for ref in references))
        scores['exact_match'] = float(np.mean(exact))
        scores['similarity'] = float(np.mean(similarity))
        return scores
