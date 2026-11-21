from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Only import Setting_Train_Only
from local_code.stage_4_code.Evaluate_Generation import Evaluate_Generation
from local_code.stage_4_code.Plot_Learning_Curve import plot_learning_curve
from local_code.stage_4_code.Text_Utils import detokenize, tokenize_joke
from difflib import SequenceMatcher
import numpy as np
import torch
import json
import os
import sys
import time

# usage: python script_rnn_generation.py [rnn] [lstm] [gru]   (no arguments = all three)
UNITS = sys.argv[1:] or ['rnn', 'lstm', 'gru']
RESULT_FOLDER = '../../result/stage_4_result/generation/'
os.makedirs(RESULT_FOLDER, exist_ok=True)

# starts that begin jokes in the dataset, and made-up starts that don't
DATASET_STARTS = ['what did the', 'why did the', 'what do you', 'did you hear', 'how do you', 'why was the',
                  'what kind of', 'knock knock .']
NEW_STARTS = ['my dog is', 'the teacher said', 'i love my', 'a computer walks', 'why do programmers']


def closest_joke(tokens, jokes):
    '''the training joke most similar to a generated one, and the word-level similarity (0-1)'''
    best = max(jokes, key=lambda ref: SequenceMatcher(None, tokens, ref).ratio())
    return best, SequenceMatcher(None, tokens, best).ratio()


#---- RNN script: joke generation from three start words ----
if 1:
    for unit in UNITS:
        #---- parameter section -------------------------------
        np.random.seed(2)
        torch.manual_seed(2)
        #------------------------------------------------------

        # ---- objection initialization setction ---------------
        data_obj = Dataset_Loader_Generation('short jokes', '')
        data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'
        data_obj.glove_file_path = '../../data/stage_4_data/glove/glove.2024.wikigiga.100d.txt'
        data = data_obj.load()

        method_obj = Method_RNN_Generation(unit.upper() + ' joke generator', '', data['embedding'], data['vocab'],
                                           rnn_type=unit)

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = RESULT_FOLDER + unit.upper() + '_'
        result_obj.result_destination_file_name = 'generation_result'

        setting_obj = Setting_Train_Only('train on all jokes, compare generations with them', '')

        evaluate_obj = Evaluate_Generation('next-word metrics, exact match, similarity', '')
        # ------------------------------------------------------

        # ---- running section ---------------------------------
        print('************ Start: %s ************' % unit.upper())
        print('device:', method_obj.device)
        # the loader is called again inside the setting; it reuses the cached GloVe rows, so this is quick
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.print_setup_summary()
        start = time.time()
        scores = setting_obj.load_run_save_evaluate()
        seconds = time.time() - start
        print('training time: %.1f s' % seconds)
        # script_generate_joke.py loads this file to write jokes from new start words without retraining
        method_obj.save_model(RESULT_FOLDER + unit.upper() + '_model.pt')
        plot_learning_curve(method_obj.history, RESULT_FOLDER + unit.upper() + '_learning_curve.png',
                            '%s joke generator: loss and next-word accuracy per epoch' % unit.upper(),
                            accuracy_label='Next-word accuracy')
        print('************ Overall Performance ************')
        for name, value in scores.items():
            print('%s %-32s %.4f' % (unit.upper(), name, value))

        # example jokes for the report, each next to the closest training joke
        jokes = data['train']['tokens']
        lines = ['%s joke generator: examples (greedy = always the most likely next word)' % unit.upper(), '']
        for title, starts in [('Starts that appear in the dataset', DATASET_STARTS),
                              ('Made-up starts (not necessarily in the dataset)', NEW_STARTS)]:
            lines.append('== ' + title + ' ==')
            for text in starts:
                tokens = method_obj.generate(tokenize_joke(text))
                ref, sim = closest_joke(tokens, jokes)
                lines.append('START:     ' + text)
                lines.append('GENERATED: ' + detokenize(tokens))
                lines.append('CLOSEST TRAINING JOKE (similarity %.2f): %s' % (sim, detokenize(ref)))
                sampled = method_obj.generate(tokenize_joke(text), temperature=0.7, seed=2)
                lines.append('SAMPLED (temperature 0.7): ' + detokenize(sampled))
                lines.append('')
        with open(RESULT_FOLDER + unit.upper() + '_examples.txt', 'w') as f:
            f.write('\n'.join(lines))
        print('\n'.join(lines))
        with open(RESULT_FOLDER + unit.upper() + '_scores.json', 'w') as f:
            json.dump({'scores': scores, 'seconds': seconds,
                       'parameters': sum(p.numel() for p in method_obj.parameters() if p.requires_grad),
                       'history': method_obj.history}, f, indent=1)
        print('************ Finish ************')
        # ------------------------------------------------------
