from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_4_code.Evaluate_Generation import Evaluate_Generation
import numpy as np
import pandas as pd
import torch
import os
import sys
import time

# Ablation studies (report section 3.7): each experiment changes ONE setting from the baseline
# (the final LSTM model of each task) and keeps the same seed, so differences come from that setting.
# The RNN and GRU rows of the report come from script_rnn_classification.py / script_rnn_generation.py.
# usage: python script_rnn_experiments.py [classification] [generation]
TASKS = sys.argv[1:] or ['classification', 'generation']
RESULT_FOLDER = '../../result/stage_4_result/experiments/'
GLOVE = '../../data/stage_4_data/glove/glove.2024.wikigiga.100d.txt'
os.makedirs(RESULT_FOLDER, exist_ok=True)

# (name, method constructor arguments, method attributes, data loader attributes)
CLASSIFICATION_EXPERIMENTS = [
    ('Baseline (LSTM)', {}, {}, {}),
    ('2 recurrent layers', {'num_layers': 2}, {}, {}),
    ('Bidirectional', {'bidirectional': True}, {}, {}),
    ('Hidden size 64', {'hidden_dim': 64}, {}, {}),
    ('Hidden size 256', {'hidden_dim': 256}, {}, {}),
    ('Frozen GloVe vectors', {'freeze_embedding': True}, {}, {}),
    ('Random word vectors (no GloVe)', {'pretrained': False}, {}, {}),
    ('No word dropout', {'embedding_dropout': 0.0}, {}, {}),
    ('Fixed learning rate (no cosine decay)', {}, {'lr_schedule': None}, {}),
    ('Keep first 256 words', {}, {}, {'keep': 'first'}),
    ('Max length 128 words', {}, {}, {'max_len': 128}),
]

GENERATION_EXPERIMENTS = [
    ('Baseline (LSTM)', {}, {}, {}),
    ('1 recurrent layer', {'num_layers': 1}, {}, {}),
    ('Hidden size 256', {'hidden_dim': 256}, {}, {}),
    ('Random word vectors (no GloVe)', {'pretrained': False}, {}, {}),
    ('Dropout 0.3', {'dropout': 0.3}, {}, {}),
]


def describe(args, attrs, loader):
    parts = ['%s=%s' % kv for kv in list(args.items()) + list(attrs.items()) + list(loader.items())]
    return ', '.join(parts) or 'baseline'


def run_classification():
    rows, cache = [], {}
    for i, (name, args, attrs, loader_attrs) in enumerate(CLASSIFICATION_EXPERIMENTS, 1):
        key = tuple(sorted(loader_attrs.items()))
        if key not in cache:
            loader = Dataset_Loader_Classification('IMDb movie reviews', '')
            loader.dataset_source_folder_path = '../../data/stage_4_data/text_classification/'
            loader.glove_file_path = GLOVE
            for k, v in loader_attrs.items():
                setattr(loader, k, v)
            cache[key] = loader.load()
        data = cache[key]
        np.random.seed(2)
        torch.manual_seed(2)
        model = Method_RNN_Classification(name, '', data['embedding'], rnn_type='lstm', **args)
        for k, v in attrs.items():
            setattr(model, k, v)
        model.track_history = False
        print('#### %d. %s (%s) on %s' % (i, name, describe(args, attrs, loader_attrs), model.device), flush=True)
        start = time.time()
        model.train(data['train']['X'], data['train']['lengths'], data['train']['y'])
        seconds = time.time() - start
        evaluator = Evaluate_Metrics('metrics', '')
        evaluator.data = {'true_y': data['test']['y'], 'pred_y': model.test(data['test']['X'], data['test']['lengths'])}
        test = evaluator.evaluate()
        evaluator.data = {'true_y': data['train']['y'],
                          'pred_y': model.test(data['train']['X'], data['train']['lengths'])}
        train = evaluator.evaluate()
        rows.append({'#': i, 'experiment': name, 'change': describe(args, attrs, loader_attrs),
                     'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                     'train_accuracy': train['accuracy'], 'test_accuracy': test['accuracy'],
                     'test_precision': test['precision'], 'test_recall': test['recall'], 'test_f1': test['f1'],
                     'seconds': round(seconds, 1)})
        print('train %.4f test %.4f (%.0f s)' % (train['accuracy'], test['accuracy'], seconds), flush=True)
    return rows


def run_generation():
    rows = []
    loader = Dataset_Loader_Generation('short jokes', '')
    loader.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'
    loader.glove_file_path = GLOVE
    data = loader.load()
    for i, (name, args, attrs, _) in enumerate(GENERATION_EXPERIMENTS, 1):
        np.random.seed(2)
        torch.manual_seed(2)
        model = Method_RNN_Generation(name, '', data['embedding'], data['vocab'], rnn_type='lstm', **args)
        for k, v in attrs.items():
            setattr(model, k, v)
        model.track_history = False
        model.data = data
        print('#### %d. %s (%s) on %s' % (i, name, describe(args, attrs, {}), model.device), flush=True)
        start = time.time()
        result = model.run()
        seconds = time.time() - start
        evaluator = Evaluate_Generation('generation', '')
        evaluator.data = result
        scores = evaluator.evaluate()
        final_loss = model.next_word_scores(data['train']['X'], data['train']['lengths'])[0]
        rows.append({'#': i, 'experiment': name, 'change': describe(args, attrs, {}),
                     'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                     'train_loss': final_loss, 'next_word_accuracy': scores['next_word_accuracy'],
                     'next_word_f1_weighted': scores['next_word_f1_weighted'],
                     'exact_match': scores['exact_match'], 'similarity': scores['similarity'],
                     'seconds': round(seconds, 1)})
        print('loss %.3f next-word acc %.4f exact match %.4f similarity %.4f (%.0f s)'
              % (final_loss, scores['next_word_accuracy'], scores['exact_match'], scores['similarity'], seconds),
              flush=True)
    return rows


if 1:
    for task in TASKS:
        rows = run_classification() if task == 'classification' else run_generation()
        table = pd.DataFrame(rows)
        base = table.iloc[0]
        metric = 'test_accuracy' if task == 'classification' else 'exact_match'
        table['change_vs_baseline'] = (table[metric] - base[metric]).round(4)
        table.to_csv(RESULT_FOLDER + task + '_summary.csv', index=False)
        print('************ %s summary ************' % task)
        print(table.drop(columns=['change']).to_string(index=False))
