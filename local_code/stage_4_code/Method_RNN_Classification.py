'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np


def fastest_device(rnn_type):
    if torch.cuda.is_available():
        return 'cuda'
    # on Apple GPUs (MPS), PyTorch's LSTM layer trained about 3x faster than on the CPU,
    # but its GRU and plain RNN layers were slower than the CPU in our timing tests, so those stay on the CPU
    if torch.backends.mps.is_available() and rnn_type == 'lstm':
        return 'mps'
    return 'cpu'


class Method_RNN_Classification(method, nn.Module):
    '''
    Recurrent network for sentiment classification of movie reviews.
    Architecture: word ids -> GloVe embedding -> recurrent layer(s) (RNN, LSTM or GRU)
                  -> hidden state after the review's last real word -> dropout -> linear -> 2 scores
    Every setting the stage 4 experiments vary is a constructor argument or class attribute.
    '''
    data = None
    max_epoch = 6
    learning_rate = 1e-3
    # the 2.8 million word-vector weights overfit quickly at the full learning rate,
    # so they get their own (smaller) one; None uses learning_rate for them too
    embedding_learning_rate = None
    # None keeps the learning rate fixed; 'cosine' lowers it smoothly to ~0 by the last epoch
    lr_schedule = 'cosine'
    batch_size = 64
    # gradient clipping: a plain RNN multiplies gradients by the same weights at every word,
    # so they can blow up on long reviews; capping their size keeps the updates stable
    clip_grad_norm = 5.0
    # 'mps' (Apple GPU), 'cuda' or 'cpu'; None picks the fastest one for the recurrent layer type
    device = None
    eval_batch_size = 500
    track_history = True
    # batches hold reviews of similar length, trimmed to a multiple of this many words,
    # so short reviews don't spend time on padding
    length_bucket = 16

    rnn_layers = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

    def __init__(self, mName, mDescription, embedding, rnn_type='lstm', hidden_dim=128, num_layers=1,
                 bidirectional=False, dropout=0.5, embedding_dropout=0.3, pretrained=True, freeze_embedding=False,
                 num_classes=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.rnn_type = rnn_type
        self.device = self.device or fastest_device(rnn_type)
        self.bidirectional = bidirectional
        vocab_size, embedding_dim = embedding.shape
        if pretrained:
            # each word starts from its GloVe vector; freeze_embedding=True keeps them fixed during training.
            # torch.tensor copies the matrix: sharing its memory would let one model's training change the
            # starting vectors of the next model trained on the same loaded data
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding), freeze=freeze_embedding,
                                                          padding_idx=0)
        else:
            # ablation: random word vectors learned from scratch
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # check here for nn.LSTM doc: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.rnn = self.rnn_layers[rnn_type](embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                                             bidirectional=bidirectional,
                                             dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        # drops whole word vectors during training, so the model can't rely on any single word
        self.embedding_dropout = embedding_dropout
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)
        self.history = {'epoch': [], 'batch_loss': [], 'train_loss': [], 'train_accuracy': [],
                        'test_loss': [], 'test_accuracy': []}

    def forward(self, x, lengths):
        '''Forward propagation: x is (batch, words) of word ids padded with 0 at the end'''
        embedded = self.embedding(x)
        if self.training and self.embedding_dropout > 0:
            keep = torch.rand(x.shape, device=x.device) >= self.embedding_dropout
            embedded = embedded * keep.unsqueeze(2) / (1 - self.embedding_dropout)
        outputs, _ = self.rnn(embedded)
        # the review's summary is the hidden state right after its last real word; later steps only saw padding.
        # (outputs at a word only depend on the words before it, so the padding never changes this state)
        last = (lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.size(2))
        summary = outputs.gather(1, last).squeeze(1)
        if self.bidirectional:
            # the backward direction reads the review from its end, so its summary sits at the first word
            half = outputs.size(2) // 2
            summary = torch.cat([summary[:, :half], outputs[:, 0, half:]], dim=1)
        return self.fc(self.dropout(summary))

    def batches(self, lengths, batch_size, shuffle):
        '''index batches of similar-length reviews (shuffled order when training)'''
        if shuffle:
            # sort by length with a little noise, so batch contents also change between epochs
            order = np.argsort(lengths + np.random.uniform(0, 8, len(lengths)))
            chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
            return [chunks[i] for i in np.random.permutation(len(chunks))]
        order = np.argsort(lengths)
        return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

    def to_device(self, X, lengths, idx):
        width = int(lengths[idx].max())
        width = min(X.shape[1], -(-width // self.length_bucket) * self.length_bucket)
        xb = torch.from_numpy(X[idx, :width]).to(self.device)
        lb = torch.from_numpy(lengths[idx]).to(self.device)
        return xb, lb

    def evaluate_split(self, X, lengths, y):
        '''mean loss, accuracy and predictions on a whole dataset, with dropout off'''
        nn.Module.train(self, False)
        preds = np.zeros(len(X), dtype='int64')
        total_loss = 0.0
        with torch.no_grad():
            for idx in self.batches(lengths, self.eval_batch_size, shuffle=False):
                xb, lb = self.to_device(X, lengths, idx)
                logits = self.forward(xb, lb)
                yb = torch.from_numpy(y[idx]).to(self.device)
                total_loss += nn.functional.cross_entropy(logits, yb, reduction='sum').item()
                preds[idx] = logits.argmax(1).cpu().numpy()
        return total_loss / len(X), float(np.mean(preds == y)), preds

    def train(self, X, lengths, y, test=None):
        self.to(self.device)
        # only parameters that require gradients (a frozen embedding is left out)
        embedding_params = [p for p in self.embedding.parameters() if p.requires_grad]
        other_params = [p for name, p in self.named_parameters() if not name.startswith('embedding.')]
        groups = [{'params': other_params}]
        if embedding_params:
            groups.append({'params': embedding_params, 'lr': self.embedding_learning_rate or self.learning_rate})
        optimizer = torch.optim.Adam(groups, lr=self.learning_rate)
        # check here for the scheduler doc: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch) if self.lr_schedule == 'cosine' else None
        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)
            epoch_loss = 0.0
            for idx in self.batches(lengths, self.batch_size, shuffle=True):
                xb, lb = self.to_device(X, lengths, idx)
                yb = torch.from_numpy(y[idx]).to(self.device)
                loss = nn.functional.cross_entropy(self.forward(xb, lb), yb)
                optimizer.zero_grad()
                # error backpropagation (through time) computes the gradients; Adam then updates the weights
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            if scheduler is not None:
                scheduler.step()
            if not self.track_history:
                print('Epoch: %d Batch Loss: %.4f' % (epoch + 1, epoch_loss / len(X)), flush=True)
                continue
            train_loss, train_acc, _ = self.evaluate_split(X, lengths, y)
            self.history['epoch'].append(epoch + 1)
            self.history['batch_loss'].append(epoch_loss / len(X))
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            message = 'Epoch: %d Train Loss: %.4f Train Accuracy: %.4f' % (epoch + 1, train_loss, train_acc)
            # test values are only recorded for the learning curves; they are never used to pick the model
            if test is not None:
                test_loss, test_acc, _ = self.evaluate_split(test['X'], test['lengths'], test['y'])
                self.history['test_loss'].append(test_loss)
                self.history['test_accuracy'].append(test_acc)
                message += ' Test Loss: %.4f Test Accuracy: %.4f' % (test_loss, test_acc)
            print(message, flush=True)

    def test(self, X, lengths):
        # each review gets the class with the larger score
        return self.evaluate_split(X, lengths, np.zeros(len(X), dtype='int64'))[2]

    def run(self):
        print('method running...')
        print('--start training...')
        train = self.data['train']
        self.train(train['X'], train['lengths'], train['y'], self.data['test'])
        print('--start testing...')
        test = self.data['test']
        pred_y = self.test(test['X'], test['lengths'])
        return {'pred_y': pred_y, 'true_y': test['y'], 'history': self.history}
