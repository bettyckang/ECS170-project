'''
Write jokes from start words you type, with a generator trained by script_rnn_generation.py.
It loads the saved model (result/stage_4_result/generation/<UNIT>_model.pt), so it needs no
retraining, no dataset and no GloVe file.

usage (from this folder, with the project root on the Python path):
    PYTHONPATH=../.. python3 script_generate_joke.py                         asks for start words, one line at a time
    PYTHONPATH=../.. python3 script_generate_joke.py "why did the" "my dog is"
    PYTHONPATH=../.. python3 script_generate_joke.py --unit gru "what do you"
'''

from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Text_Utils import detokenize, tokenize_joke, UNK
import argparse
import os

MODEL_FOLDER = '../../result/stage_4_result/generation/'


def write_joke(model, text, temperature):
    tokens = tokenize_joke(text)
    if not tokens:
        return
    unknown = [w for w in tokens if w not in model.index]
    print('START:     ' + text)
    if unknown:
        print('           (never seen in training, read as %s: %s)' % (UNK, ', '.join(unknown)))
    print('GENERATED: ' + detokenize(model.generate(tokens)))
    if temperature > 0:
        sampled = model.generate(tokens, temperature=temperature)
        print('SAMPLED (temperature %.1f): %s' % (temperature, detokenize(sampled)))
    print()


#---- RNN script: write jokes from typed start words with a saved generator ----
if 1:
    parser = argparse.ArgumentParser(description='Finish jokes from start words with a trained RNN generator.')
    parser.add_argument('starts', nargs='*', help='start words, e.g. "why did the" (the task uses three words)')
    parser.add_argument('--unit', default='lstm', choices=['rnn', 'lstm', 'gru'], help='which saved model to use')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='also print a sampled joke at this temperature (0 = only the greedy joke)')
    args = parser.parse_args()

    model_path = MODEL_FOLDER + args.unit.upper() + '_model.pt'
    if not os.path.exists(model_path):
        raise SystemExit('no saved model at %s: run script_rnn_generation.py %s first' % (model_path, args.unit))
    model = Method_RNN_Generation.load_model(model_path)
    print('loaded %s (%d words in the vocabulary)' % (model_path, len(model.vocab)))
    print('GENERATED always picks the most likely next word; SAMPLED picks at random by probability.\n')

    if args.starts:
        for text in args.starts:
            write_joke(model, text, args.temperature)
    else:
        while True:
            try:
                text = input('start words (Enter on an empty line to quit): ').strip()
            except EOFError:
                break
            if not text:
                break
            write_joke(model, text, args.temperature)
