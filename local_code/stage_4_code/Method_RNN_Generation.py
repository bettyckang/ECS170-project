'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_4_code.Text_Utils import UNK, EOS
from local_code.stage_4_code.Method_RNN_Classification import fastest_device
import torch
from torch import nn
import numpy as np


class Method_RNN_Generation(method, nn.Module):
    '''
    Word-level language model that writes the rest of a joke from its first three words.
    Architecture: word ids -> GloVe embedding -> recurrent layers (RNN, LSTM or GRU)
                  -> dropout -> linear -> one score per vocabulary word (the next word)
    Training: at every position the model sees the joke so far and is scored on predicting the next word
    (teacher forcing). Generation: feed the three start words, then repeatedly append the
    highest-scoring next word until the model predicts <eos> (end of joke).
    '''
    data = None
    max_epoch = 40
    learning_rate = 3e-3
    batch_size = 32
    clip_grad_norm = 5.0
    # 'mps' (Apple GPU), 'cuda' or 'cpu'; None picks the fastest one for the recurrent layer type
    device = None
    eval_batch_size = 512
    track_history = True
    # the task gives the model a joke's first three words
    start_words = 3
    # stop generating after this many words if the model never predicts <eos>
    max_generate_len = 60

    rnn_layers = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

    def __init__(self, mName, mDescription, embedding, vocab, rnn_type='lstm', hidden_dim=512, num_layers=2,
                 dropout=0.0, pretrained=True, freeze_embedding=False):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.rnn_type = rnn_type
        self.device = self.device or fastest_device(rnn_type)
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        vocab_size, embedding_dim = embedding.shape
        if pretrained:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding), freeze=freeze_embedding,
                                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = self.rnn_layers[rnn_type](embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                                             dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.history = {'epoch': [], 'batch_loss': [], 'train_loss': [], 'train_accuracy': [],
                        'test_loss': [], 'test_accuracy': []}

    def forward(self, x, hidden=None):
        '''Forward propagation: scores for the next word at every position, plus the recurrent state'''
        outputs, hidden = self.rnn(self.embedding(x), hidden)
        return self.fc(self.dropout(outputs)), hidden

    def next_word_scores(self, X, lengths, from_position=0):
        '''
        teacher-forced next-word predictions on whole jokes, with dropout off.
        Returns mean loss and accuracy over the predicted words, plus the true and predicted word ids
        for every target position >= from_position (3 = only the words after the three given ones).
        '''
        nn.Module.train(self, False)
        total_loss, count, true_ids, pred_ids = 0.0, 0, [], []
        with torch.no_grad():
            for start in range(0, len(X), self.eval_batch_size):
                width = int(lengths[start:start + self.eval_batch_size].max())
                xb = torch.from_numpy(X[start:start + self.eval_batch_size, :width]).to(self.device)
                logits, _ = self.forward(xb[:, :-1])
                target = xb[:, 1:].clone()
                # target position t predicts word t+1 of the joke; skip padding and the first words if asked
                target[:, :max(0, from_position - 1)] = 0
                mask = target != 0
                total_loss += nn.functional.cross_entropy(logits[mask], target[mask], reduction='sum').item()
                count += int(mask.sum())
                true_ids.append(target[mask].cpu().numpy())
                pred_ids.append(logits[mask].argmax(1).cpu().numpy())
        true_ids, pred_ids = np.concatenate(true_ids), np.concatenate(pred_ids)
        return total_loss / count, float(np.mean(true_ids == pred_ids)), true_ids, pred_ids

    def train(self, X, lengths):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        n = len(X)
        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)
            permutation = np.random.permutation(n)
            epoch_loss, epoch_words = 0.0, 0
            for start in range(0, n, self.batch_size):
                idx = permutation[start:start + self.batch_size]
                width = int(lengths[idx].max())
                xb = torch.from_numpy(X[idx, :width]).to(self.device)
                logits, _ = self.forward(xb[:, :-1])
                target = xb[:, 1:]
                # padding positions (word id 0) don't count toward the loss
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(2)), target.reshape(-1),
                                                   ignore_index=0)
                optimizer.zero_grad()
                # error backpropagation through time, then an Adam update
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
                optimizer.step()
                words = int((target != 0).sum())
                epoch_loss += loss.item() * words
                epoch_words += words
            if not self.track_history:
                if (epoch + 1) % 10 == 0:
                    print('Epoch: %d Batch Loss: %.4f' % (epoch + 1, epoch_loss / epoch_words), flush=True)
                continue
            train_loss, train_acc, _, _ = self.next_word_scores(X, lengths)
            self.history['epoch'].append(epoch + 1)
            self.history['batch_loss'].append(epoch_loss / epoch_words)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            print('Epoch: %d Train Loss: %.4f Next-word Accuracy: %.4f' % (epoch + 1, train_loss, train_acc),
                  flush=True)

    def generate_many(self, starts, temperature=0.0, seed=None):
        '''
        continue several jokes at once from start-token lists of equal length.
        temperature 0 always picks the most likely next word (greedy); a temperature above 0 samples,
        so the same start can give different jokes. Words the model never saw become <unk>.
        '''
        nn.Module.train(self, False)
        self.to(self.device)
        rng = np.random.RandomState(seed)
        unk, eos = self.index[UNK], self.index[EOS]
        ids = torch.tensor([[self.index.get(w, unk) for w in start] for start in starts], device=self.device)
        outputs = [list(start) for start in starts]
        finished = np.zeros(len(starts), dtype=bool)
        with torch.no_grad():
            logits, hidden = self.forward(ids)
            for _ in range(self.max_generate_len):
                scores = logits[:, -1]
                if temperature > 0:
                    probs = torch.softmax(scores / temperature, dim=1).cpu().numpy().astype('float64')
                    next_ids = np.array([rng.choice(probs.shape[1], p=p / p.sum()) for p in probs])
                else:
                    next_ids = scores.argmax(1).cpu().numpy()
                for i, next_id in enumerate(next_ids):
                    if finished[i]:
                        continue
                    if next_id == eos:
                        finished[i] = True
                    else:
                        outputs[i].append(self.vocab[next_id])
                if finished.all():
                    break
                logits, hidden = self.forward(torch.from_numpy(next_ids).view(-1, 1).to(self.device), hidden)
        return outputs

    def generate(self, start_tokens, temperature=0.0, seed=None):
        '''continue one joke from its start tokens'''
        return self.generate_many([list(start_tokens)], temperature, seed)[0]

    def save_model(self, path):
        '''
        save the trained weights with the vocabulary and layer sizes, so jokes can be generated later
        without retraining (the fine-tuned word vectors are in the weights, so GloVe isn't needed either)
        '''
        torch.save({'rnn_type': self.rnn_type, 'hidden_dim': self.rnn.hidden_size, 'num_layers': self.rnn.num_layers,
                    'vocab': list(self.vocab),
                    'state_dict': {k: v.detach().cpu() for k, v in self.state_dict().items()}}, path)
        print('saved model to', path)

    @classmethod
    def load_model(cls, path):
        '''rebuild a generator saved by save_model'''
        saved = torch.load(path, map_location='cpu', weights_only=True)
        vocab = saved['vocab']
        # placeholder vectors of the right shape; load_state_dict replaces them with the trained ones
        embedding = np.zeros((len(vocab), saved['state_dict']['embedding.weight'].shape[1]), dtype=np.float32)
        model = cls(saved['rnn_type'].upper() + ' joke generator', 'loaded from ' + path, embedding, vocab,
                    rnn_type=saved['rnn_type'], hidden_dim=saved['hidden_dim'], num_layers=saved['num_layers'])
        model.load_state_dict(saved['state_dict'])
        return model

    def run(self):
        print('method running...')
        print('--start training...')
        train = self.data['train']
        self.train(train['X'], train['lengths'])
        print('--start generating...')
        # one greedy joke for every distinct three-word start in the training data
        starts = sorted({tuple(tokens[:self.start_words]) for tokens in train['tokens']
                         if len(tokens) > self.start_words})
        generated = dict(zip(starts, self.generate_many([list(start) for start in starts])))
        # next-word predictions for the words after the three given ones, for accuracy/precision/recall/F1
        _, _, true_ids, pred_ids = self.next_word_scores(train['X'], train['lengths'], from_position=self.start_words)
        return {'generated': generated, 'jokes': train['tokens'], 'true_y': true_ids, 'pred_y': pred_ids,
                'history': self.history}
