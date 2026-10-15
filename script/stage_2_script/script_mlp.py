from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test_Given import Setting_Train_Test_Given
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Plot_Learning_Curve import plot_learning_curve
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('stage 2 digits', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_train_file_name = 'train.csv'
    data_obj.dataset_test_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '', hidden_dims=(256, 128), activation='relu', dropout=0.2)
    method_obj.max_epoch = 30
    method_obj.learning_rate = 1e-3
    method_obj.batch_size = 128
    method_obj.optimizer_name = 'adam'
    method_obj.loss_name = 'cross_entropy'

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Given('pre-partitioned train test', '')

    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores = setting_obj.load_run_save_evaluate()
    plot_learning_curve(method_obj.history, '../../result/stage_2_result/MLP_learning_curve.png')
    print('************ Overall Performance ************')
    for name, value in scores.items():
        print('MLP ' + name + ': %.4f' % value)
    print('************ Finish ************')
    # ------------------------------------------------------
