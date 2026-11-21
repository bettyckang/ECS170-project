from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Test_Given import Setting_Train_Test_Given
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_4_code.Plot_Learning_Curve import plot_learning_curve
import numpy as np
import torch
import json
import os
import sys
import time

# usage: python script_rnn_classification.py [rnn] [lstm] [gru]   (no arguments = all three)
UNITS = sys.argv[1:] or ['rnn', 'lstm', 'gru']
RESULT_FOLDER = '../../result/stage_4_result/classification/'
os.makedirs(RESULT_FOLDER, exist_ok=True)


class Cached_Loader(Dataset_Loader_Classification):
    # cleaning 50,000 reviews takes a while, so the three models share one loaded copy
    cache = None

    def load(self):
        if Cached_Loader.cache is None:
            Cached_Loader.cache = super().load()
        return Cached_Loader.cache


#---- RNN script: IMDb movie review sentiment classification ----
if 1:
    data_obj = Cached_Loader('IMDb movie reviews', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/'
    data_obj.glove_file_path = '../../data/stage_4_data/glove/glove.2024.wikigiga.100d.txt'
    data = data_obj.load()

    for unit in UNITS:
        #---- parameter section -------------------------------
        np.random.seed(2)
        torch.manual_seed(2)
        #------------------------------------------------------

        # ---- objection initialization setction ---------------
        method_obj = Method_RNN_Classification(unit.upper() + ' sentiment classifier', '', data['embedding'],
                                               rnn_type=unit)

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = RESULT_FOLDER + unit.upper() + '_'
        result_obj.result_destination_file_name = 'prediction_result'

        setting_obj = Setting_Train_Test_Given('pre-partitioned train test', '')

        evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')
        # ------------------------------------------------------

        # ---- running section ---------------------------------
        print('************ Start: %s ************' % unit.upper())
        print('device:', method_obj.device)
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.print_setup_summary()
        start = time.time()
        test_scores = setting_obj.load_run_save_evaluate()
        seconds = time.time() - start
        print('training time: %.1f s' % seconds)
        plot_learning_curve(method_obj.history, RESULT_FOLDER + unit.upper() + '_learning_curve.png',
                            '%s on IMDb reviews: loss and accuracy per epoch' % unit.upper())
        # the same metrics on the training set, for the report's train/test comparison
        train = data['train']
        evaluate_obj.data = {'true_y': train['y'], 'pred_y': method_obj.test(train['X'], train['lengths'])}
        train_scores = evaluate_obj.evaluate()
        print('************ Overall Performance ************')
        for name in test_scores:
            print('%s %-18s train: %.4f  test: %.4f' % (unit.upper(), name, train_scores[name], test_scores[name]))
        with open(RESULT_FOLDER + unit.upper() + '_scores.json', 'w') as f:
            json.dump({'train': train_scores, 'test': test_scores, 'seconds': seconds,
                       'parameters': sum(p.numel() for p in method_obj.parameters() if p.requires_grad),
                       'history': method_obj.history}, f, indent=1)
        print('************ Finish ************')
        # ------------------------------------------------------
