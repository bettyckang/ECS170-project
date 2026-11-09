'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np


class Method_CNN(method, nn.Module):
    '''
    Shared CNN used by Method_CNN_MNIST, Method_CNN_ORL and Method_CNN_CIFAR.
    Architecture: [conv -> (batch norm) -> activation -> pool] x len(conv_channels) -> flatten -> fc -> dropout -> fc
    Every setting the stage 3 ablations vary (depth, kernel size, padding, stride, pooling,
    hidden layer dimension, loss) is a constructor argument or class attribute.
    '''
    data = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 128
    # one of: 'adam', 'adamw', 'sgd', 'rmsprop'
    optimizer_name = 'adam'
    # one of: 'cross_entropy', 'nll', 'multi_margin'
    loss_name = 'cross_entropy'
    # None keeps the learning rate fixed; 'cosine' lowers it smoothly to ~0 by the last epoch,
    # so training settles at the end instead of stopping in the middle of a random swing
    lr_schedule = None
    # 'mps' (Apple GPU), 'cuda' or 'cpu'; defaults to the fastest one available
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    # batch size used when scoring a whole dataset (no gradients, so it can be large)
    eval_batch_size = 1000
    # score both full datasets after every epoch for the learning curves; the configuration
    # experiments turn this off to save time, since they only need the final scores
    track_history = True

    activation_funcs = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'leaky_relu': nn.LeakyReLU}
    pool_layers = {'max': nn.MaxPool2d, 'avg': nn.AvgPool2d}

    def __init__(self, mName, mDescription, in_channels, image_size, num_classes, conv_channels=(32, 64),
                 kernel_size=3, padding=1, stride=1, pool='max', hidden_dim=128, dropout=0.25,
                 activation='relu', batch_norm=False):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        layers = []
        prev_channels = in_channels
        for channels in conv_channels:
            # check here for nn.Conv2d doc: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            layers.append(nn.Conv2d(prev_channels, channels, kernel_size=kernel_size, stride=stride, padding=padding))
            if batch_norm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(self.activation_funcs[activation]())
            if pool is not None:
                # 2x2 pooling halves the height and width of each feature map
                layers.append(self.pool_layers[pool](kernel_size=2))
            prev_channels = channels
        self.features = nn.Sequential(*layers)
        # run one dummy image through the conv layers to find how many values the classifier receives
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, in_channels, *image_size)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            self.activation_funcs[activation](),
            nn.Dropout(dropout),
            # raw scores (logits); cross-entropy applies softmax itself
            nn.Linear(hidden_dim, num_classes),
        )
        # one entry per epoch, used for the learning curves; train/test values are measured
        # on the full datasets after each epoch with dropout off
        self.history = {'epoch': [], 'batch_loss': [], 'train_loss': [], 'train_accuracy': [],
                        'test_loss': [], 'test_accuracy': []}

    def forward(self, x):
        '''Forward propagation'''
        return self.classifier(self.features(x))

    def get_optimizer(self):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.optimizer_name == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        if self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        raise ValueError('unknown optimizer: ' + self.optimizer_name)

    def compute_loss(self, logits, y_true, reduction='mean'):
        if self.loss_name == 'cross_entropy':
            return nn.functional.cross_entropy(logits, y_true, reduction=reduction)
        if self.loss_name == 'nll':
            return nn.functional.nll_loss(nn.functional.log_softmax(logits, dim=1), y_true, reduction=reduction)
        if self.loss_name == 'multi_margin':
            # multi-class hinge loss, the SVM-style alternative. PyTorch's multi_margin_loss is not
            # implemented on Apple GPUs (MPS), so we compute the same formula directly:
            # for each wrong class i, max(0, 1 - score[correct] + score[i]), averaged over the classes
            correct = logits.gather(1, y_true.unsqueeze(1))
            margins = (1 - correct + logits).clamp(min=0)
            margins = margins.scatter(1, y_true.unsqueeze(1), 0)
            per_instance = margins.sum(1) / logits.shape[1]
            return per_instance.sum() if reduction == 'sum' else per_instance.mean()
        raise ValueError('unknown loss: ' + self.loss_name)

    def evaluate_split(self, X, y):
        # loss, accuracy and predictions on a whole dataset, with dropout and batch-norm updates off
        # (nn.Module.train is called directly because this class overrides train() with the training loop)
        nn.Module.train(self, False)
        total_loss, preds = 0.0, []
        with torch.no_grad():
            for start in range(0, len(X), self.eval_batch_size):
                xb = torch.from_numpy(X[start:start + self.eval_batch_size]).to(self.device)
                yb = torch.from_numpy(y[start:start + self.eval_batch_size]).to(self.device)
                logits = self.forward(xb)
                total_loss += self.compute_loss(logits, yb, reduction='sum').item()
                preds.append(logits.argmax(1).cpu().numpy())
        preds = np.concatenate(preds)
        return total_loss / len(X), float(np.mean(preds == y)), preds

    def train(self, X, y, test_X=None, test_y=None):
        self.to(self.device)
        optimizer = self.get_optimizer()
        # check here for the scheduler doc: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch) if self.lr_schedule == 'cosine' else None
        n = len(X)
        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)
            # shuffle the training set each epoch, then walk through it batch by batch
            permutation = np.random.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = permutation[start:start + self.batch_size]
                xb = torch.from_numpy(X[idx]).to(self.device)
                yb = torch.from_numpy(y[idx]).to(self.device)
                loss = self.compute_loss(self.forward(xb), yb)
                optimizer.zero_grad()
                # error backpropagation computes the gradients; the optimizer then updates the weights
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            if scheduler is not None:
                scheduler.step()

            if not self.track_history:
                print('Epoch: %d Batch Loss: %.4f' % (epoch + 1, epoch_loss / n), flush=True)
                continue
            train_loss, train_acc, _ = self.evaluate_split(X, y)
            self.history['epoch'].append(epoch + 1)
            self.history['batch_loss'].append(epoch_loss / n)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            message = 'Epoch: %d Train Loss: %.4f Train Accuracy: %.4f' % (epoch + 1, train_loss, train_acc)
            # test values are only recorded for the learning curves; they are never used to pick the model
            if test_X is not None:
                test_loss, test_acc, _ = self.evaluate_split(test_X, test_y)
                self.history['test_loss'].append(test_loss)
                self.history['test_accuracy'].append(test_acc)
                message += ' Test Loss: %.4f Test Accuracy: %.4f' % (test_loss, test_acc)
            print(message, flush=True)

    def test(self, X):
        # instances get the label with the largest score
        return self.evaluate_split(X, np.zeros(len(X), dtype='int64'))[2]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'],
                   self.data['test']['X'], self.data['test']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'history': self.history}
