from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Given import Setting_Train_Test_Given
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Plot_Learning_Curve import plot_learning_curve
import numpy as np
import torch
import time

#---- CNN script: CIFAR-10 colored objects ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('CIFAR', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'CIFAR'

    method_obj = Method_CNN_CIFAR('convolutional neural network', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_CIFAR_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Given('pre-partitioned train test', '')

    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    print('device:', method_obj.device)
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    start = time.time()
    test_scores = setting_obj.load_run_save_evaluate()
    print('training time: %.1f s' % (time.time() - start))
    plot_learning_curve(method_obj.history, '../../result/stage_3_result/CNN_CIFAR_learning_curve.png',
                        'CNN on CIFAR: loss and accuracy per epoch')
    # the same metrics on the training set, for the report's train/test comparison
    evaluate_obj.data = {'true_y': method_obj.data['train']['y'], 'pred_y': method_obj.test(method_obj.data['train']['X'])}
    train_scores = evaluate_obj.evaluate()
    print('************ Overall Performance ************')
    for name in test_scores:
        print('CNN CIFAR %-20s train: %.4f  test: %.4f' % (name, train_scores[name], test_scores[name]))
    print('************ Finish ************')
    # ------------------------------------------------------
