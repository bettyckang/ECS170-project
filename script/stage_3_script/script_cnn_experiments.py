from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import pandas as pd
import torch
import time
import sys
import os

#---- CNN configuration experiments for step 3-5: change one setting at a time from each dataset's baseline ----
# usage: python script_cnn_experiments.py [MNIST] [ORL] [CIFAR]   (no arguments = all three)
if 1:
    #---- parameter section -------------------------------
    # shorter training than the final models (MNIST 10, CIFAR 20 epochs) to keep 30 runs affordable;
    # each dataset's baseline is retrained with the same budget, so the comparison is fair
    datasets = {
        'MNIST': {'method': Method_CNN_MNIST, 'epochs': 5,
                  'fewer': (32,), 'more': (32, 64, 128)},
        'ORL': {'method': Method_CNN_ORL, 'epochs': 40,
                'fewer': (16, 32), 'more': (16, 32, 64, 128)},
        'CIFAR': {'method': Method_CNN_CIFAR, 'epochs': 10,
                  'fewer': (32, 64), 'more': (32, 64, 128, 256)},
    }
    def experiments(d):
        return [
            ('Baseline', {}, {}),
            ('Depth: one fewer conv block', {'conv_channels': d['fewer']}, {}),
            ('Depth: one more conv block', {'conv_channels': d['more']}, {}),
            ('Kernel size 5x5 (padding 2)', {'kernel_size': 5, 'padding': 2}, {}),
            ('No padding (3x3 kernel)', {'padding': 0}, {}),
            ('Stride 2 instead of pooling', {'stride': 2, 'pool': None}, {}),
            ('Average pooling', {'pool': 'avg'}, {}),
            ('Hidden layer 32', {'hidden_dim': 32}, {}),
            ('Hidden layer 512', {'hidden_dim': 512}, {}),
            ('Multi-margin (hinge) loss', {}, {'loss_name': 'multi_margin'}),
        ]
    selected = sys.argv[1:] or list(datasets)
    result_folder = '../../result/stage_3_result/experiments/'
    os.makedirs(result_folder, exist_ok=True)
    #------------------------------------------------------

    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')
    for name in selected:
        d = datasets[name]
        data_obj = Dataset_Loader(name, '')
        data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
        data_obj.dataset_source_file_name = name
        loaded_data = data_obj.load()

        rows = []
        for i, (label, arch_changes, train_changes) in enumerate(experiments(d)):
            print('************ %s experiment %d: %s ************' % (name, i + 1, label), flush=True)
            # same seed for every run so differences come from the setting, not the initialization
            np.random.seed(2)
            torch.manual_seed(2)
            method_obj = d['method']('convolutional neural network', label, **arch_changes)
            method_obj.max_epoch = d['epochs']
            method_obj.track_history = False
            for key, value in train_changes.items():
                setattr(method_obj, key, value)
            method_obj.data = loaded_data

            start = time.time()
            result = method_obj.run()
            seconds = time.time() - start

            evaluate_obj.data = result
            test_scores = evaluate_obj.evaluate()
            evaluate_obj.data = {'true_y': loaded_data['train']['y'], 'pred_y': method_obj.test(loaded_data['train']['X'])}
            train_scores = evaluate_obj.evaluate()
            rows.append({'#': i + 1, 'experiment': label,
                         'parameters': sum(p.numel() for p in method_obj.parameters()),
                         'train_accuracy': train_scores['accuracy'], 'test_accuracy': test_scores['accuracy'],
                         'test_precision_macro': test_scores['precision_macro'],
                         'test_recall_macro': test_scores['recall_macro'], 'test_f1_macro': test_scores['f1_macro'],
                         'seconds': round(seconds, 1)})
            print('train %.4f  test %.4f  (%.1f s)' % (train_scores['accuracy'], test_scores['accuracy'], seconds), flush=True)

        #---- summary section ---------------------------------
        df = pd.DataFrame(rows)
        df['change_vs_baseline'] = df['test_accuracy'] - df.loc[0, 'test_accuracy']
        df.to_csv(result_folder + name + '_summary.csv', index=False)
        print('************ %s summary (%d epochs) ************' % (name, d['epochs']))
        print(df[['#', 'experiment', 'parameters', 'train_accuracy', 'test_accuracy', 'test_f1_macro',
                  'change_vs_baseline', 'seconds']].to_string(index=False, float_format='%.4f'), flush=True)
        #------------------------------------------------------
