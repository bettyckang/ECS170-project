'''
Text cleaning, vocabulary and GloVe helpers shared by the two stage 4 dataset loaders
'''

import os
import re
import hashlib
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PAD, UNK, EOS = '<pad>', '<unk>', '<eos>'

# scikit-learn's English stop-word list, minus the words that carry sentiment.
# "not good" and "good" mean opposite things, so negations, contrast words ("but", "however")
# and intensity words ("very", "too", "less") must stay in the reviews.
KEEP_WORDS = {'no', 'not', 'nor', 'never', 'nothing', 'none', 'nobody', 'cannot', 'but', 'however',
              'although', 'though', 'yet', 'very', 'too', 'well', 'least', 'less', 'more', 'most',
              'few', 'much', 'enough', 'only', 'against', 'serious'}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - KEEP_WORDS


def clean_review(text):
    '''movie review -> list of lower-case words, without HTML, punctuation, numbers or stop words'''
    text = text.lower().replace('<br />', ' ')
    # expand negative contractions so "didn't" keeps its meaning as "did not"
    text = text.replace("won't", 'will not').replace("can't", 'can not').replace("n't", ' not')
    # keep letters only; other contractions ('s, 're, 'll, 've, 'm, 'd) become separate stop words
    words = re.findall(r'[a-z]+', text)
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


# joke tokens: punctuation stays (a joke's "?" and "." matter) and contractions are split the same
# way GloVe splits them ("didn't" -> "did" + "n't", "what's" -> "what" + "'s")
JOKE_TOKEN = re.compile(r"[a-z0-9]+(?=n't\b)|n't\b|'(?:s|re|ve|ll|d|m)\b|[a-z0-9]+|[^\sa-z0-9]")


def tokenize_joke(text):
    text = text.lower().replace('’', "'").replace('“', '"').replace('”', '"')
    return JOKE_TOKEN.findall(text)


ATTACH_LEFT = re.compile(r"n't|'(?:s|re|ve|ll|d|m)|[.,!?;:%)]")


def detokenize(tokens):
    '''tokens -> readable text: no space before punctuation or split contractions, quotes hug their words'''
    text, glue_next, quote_open = '', False, {'"': False, "'": False}
    for tok in tokens:
        if tok in quote_open and quote_open[tok]:
            # closing quote: attach to the word before it
            text, quote_open[tok], glue_next = text + tok, False, False
            continue
        glue = glue_next or not text or ATTACH_LEFT.fullmatch(tok) or tok == '-'
        text += tok if glue else ' ' + tok
        if tok in quote_open:
            quote_open[tok] = True
        glue_next = tok in quote_open or tok in ('(', '$', '-')
    return text


def build_vocabulary(token_lists, min_count=1, specials=(PAD, UNK)):
    '''word list with the special tokens first (so <pad> is index 0), then words by frequency'''
    counts = Counter(w for tokens in token_lists for w in tokens)
    words = sorted((w for w, c in counts.items() if c >= min_count), key=lambda w: (-counts[w], w))
    return list(specials) + words


def load_glove(glove_path, vocab, dim=100, seed=2):
    '''
    embedding matrix (len(vocab) x dim) initialized from GloVe.
    words GloVe doesn't know get small random vectors; <pad> is all zeros.
    The full GloVe file is 1.6 GB, so the rows we need are cached in a small text file next to it
    (named by a hash of the vocabulary, so a different vocabulary builds a new cache).
    '''
    key = hashlib.md5('\n'.join(vocab).encode('utf-8')).hexdigest()[:10]
    cache_path = os.path.join(os.path.dirname(glove_path), 'cache_%dd_%s.txt' % (dim, key))
    wanted = set(vocab)
    vectors = {}
    source = cache_path if os.path.exists(cache_path) else glove_path
    with open(source, encoding='utf-8') as f:
        for line in f:
            word, rest = line.rstrip('\n').split(' ', 1)
            if word in wanted:
                vectors[word] = np.array(rest.split(' '), dtype='float32')
    if source == glove_path:
        with open(cache_path, 'w', encoding='utf-8') as f:
            for word, vec in vectors.items():
                f.write(word + ' ' + ' '.join('%.6f' % v for v in vec) + '\n')

    rng = np.random.RandomState(seed)
    known = np.stack(list(vectors.values())) if vectors else np.zeros((1, dim), dtype='float32')
    # random vectors with the same spread as the real GloVe vectors
    matrix = rng.normal(0, known.std(), size=(len(vocab), dim)).astype('float32')
    for i, word in enumerate(vocab):
        if word in vectors:
            matrix[i] = vectors[word]
    matrix[vocab.index(PAD)] = 0
    coverage = sum(1 for w in vocab if w in vectors) / len(vocab)
    return matrix, coverage
