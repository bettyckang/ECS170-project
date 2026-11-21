from local_code.stage_4_code.Plot_Learning_Curve import plot_loss_comparison
import json

# report section 3.5: training loss per epoch for RNN, LSTM and GRU on both datasets
# (run script_rnn_classification.py and script_rnn_generation.py first)
RESULT = '../../result/stage_4_result/'


def histories(task):
    out = {}
    for unit in ['RNN', 'LSTM', 'GRU']:
        with open(RESULT + task + '/' + unit + '_scores.json') as f:
            out[unit] = json.load(f)['history']
    return out


if 1:
    plot_loss_comparison([('Text classification (IMDb reviews)', histories('classification')),
                          ('Text generation (jokes)', histories('generation'))],
                         RESULT + 'training_loss_all_models.png')
