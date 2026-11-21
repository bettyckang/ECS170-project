'''
Concrete Evaluate class for classification metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Metrics(evaluate):
    data = None
    # 'binary' scores the positive class (sentiment: positive review = 1);
    # 'macro' averages the classes equally; 'weighted' by how often each class occurs
    averages = ('binary', 'macro')

    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        scores = {'accuracy': accuracy_score(true_y, pred_y)}
        for average in self.averages:
            suffix = '' if average == 'binary' else '_' + average
            scores['precision' + suffix] = precision_score(true_y, pred_y, average=average, zero_division=0)
            scores['recall' + suffix] = recall_score(true_y, pred_y, average=average, zero_division=0)
            scores['f1' + suffix] = f1_score(true_y, pred_y, average=average, zero_division=0)
        return scores
