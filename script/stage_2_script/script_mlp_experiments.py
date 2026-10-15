from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Plot_Learning_Curve import plot_learning_curve
import numpy as np
import pandas as pd
import torch
import time
import os

#---- MLP experiments for step 2-4: change one setting at a time from the baseline ----
if 1:
    #---- parameter section -------------------------------
    max_epoch = 20
    baseline = {'hidden_dims': (256, 128), 'activation': 'relu', 'dropout': 0.2,
                'optimizer_name': 'adam', 'learning_rate': 1e-3, 'loss_name': 'cross_entropy', 'device': 'cpu'}
    experiments = [
        ('baseline', {}),
        ('1 hidden layer (128)', {'hidden_dims': (128,)}),
        ('3 hidden layers (512-256-128)', {'hidden_dims': (512, 256, 128)}),
        ('no dropout', {'dropout': 0.0}),
        ('tanh activation', {'activation': 'tanh'}),
        ('sigmoid activation', {'activation': 'sigmoid'}),
        ('SGD + momentum, lr 0.01', {'optimizer_name': 'sgd', 'learning_rate': 1e-2}),
        ('RMSprop', {'optimizer_name': 'rmsprop'}),
        ('AdamW', {'optimizer_name': 'adamw'}),
        ('Adam, lr 0.01', {'learning_rate': 1e-2}),
        ('NLL loss', {'loss_name': 'nll'}),
        ('multi-margin (hinge) loss', {'loss_name': 'multi_margin'}),
        ('baseline on Apple GPU (MPS)', {'device': 'mps'}),
    ]
    result_folder = '../../result/stage_2_result/experiments/'
    os.makedirs(result_folder, exist_ok=True)
    #------------------------------------------------------

    data_obj = Dataset_Loader('stage 2 digits', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_train_file_name = 'train.csv'
    data_obj.dataset_test_file_name = 'test.csv'
    loaded_data = data_obj.load()

    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')

    rows = []
    for i, (name, changes) in enumerate(experiments):
        if changes.get('device') == 'mps' and not torch.backends.mps.is_available():
            print('skipping', name, '(MPS not available)')
            continue
        config = {**baseline, **changes}
        print('************ Experiment %d: %s ************' % (i + 1, name))
        # same seed for every run so the differences come from the setting, not the initialization
        np.random.seed(2)
        torch.manual_seed(2)

        method_obj = Method_MLP('multi-layer perceptron', name, hidden_dims=config['hidden_dims'],
                                activation=config['activation'], dropout=config['dropout'])
        method_obj.max_epoch = max_epoch
        method_obj.optimizer_name = config['optimizer_name']
        method_obj.learning_rate = config['learning_rate']
        method_obj.loss_name = config['loss_name']
        method_obj.device = config['device']
        method_obj.data = loaded_data

        start = time.time()
        learned_result = method_obj.run()
        train_seconds = time.time() - start

        evaluate_obj.data = learned_result
        scores = evaluate_obj.evaluate()
        plot_learning_curve(method_obj.history, result_folder + 'curve_%02d.png' % (i + 1), title=name)
        rows.append({'#': i + 1, 'experiment': name, **scores,
                     'final_train_loss': method_obj.history['train_loss'][-1], 'seconds': round(train_seconds, 1)})

    #---- summary section ---------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(result_folder + 'summary.csv', index=False)
    print('************ Summary ************')
    print(df[['#', 'experiment', 'accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
              'f1_macro', 'seconds']].to_string(index=False, float_format='%.4f'))
    #------------------------------------------------------
