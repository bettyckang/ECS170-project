'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 30
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    # unlike stage 1, 60k instances is too much for one full batch, so we train with mini-batches
    batch_size = 128
    # one of: 'adam', 'adamw', 'sgd', 'rmsprop'
    optimizer_name = 'adam'
    # one of: 'cross_entropy', 'nll', 'multi_margin'
    loss_name = 'cross_entropy'
    # 'cpu' or 'mps' (Apple GPU) or 'cuda'
    device = 'cpu'

    activation_funcs = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'leaky_relu': nn.LeakyReLU}

    # it defines the the MLP model architecture
    # the input portal is 784 (28x28 pixels) and the output portal is 10 (digit classes 0-9)
    # hidden_dims lists the size of each hidden layer, so (256, 128) means 784 -> 256 -> 128 -> 10
    def __init__(self, mName, mDescription, input_dim=784, hidden_dims=(256, 128), output_dim=10,
                 activation='relu', dropout=0.2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(self.activation_funcs[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        # filled in during training: one entry per epoch, used for the learning curve plots
        self.history = {'epoch': [], 'train_loss': [], 'train_accuracy': [], 'test_accuracy': []}

    # it defines the forward propagation function for input x
    # note: unlike the stage 1 template, we return raw logits without a softmax,
    # because nn.CrossEntropyLoss already applies log-softmax internally (softmax twice slows learning)
    def forward(self, x):
        '''Forward propagation'''
        return self.layers(x)

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

    def compute_loss(self, logits, y_true):
        if self.loss_name == 'cross_entropy':
            # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            return nn.functional.cross_entropy(logits, y_true)
        if self.loss_name == 'nll':
            return nn.functional.nll_loss(nn.functional.log_softmax(logits, dim=1), y_true)
        if self.loss_name == 'multi_margin':
            # multi-class hinge loss, the SVM-style alternative
            return nn.functional.multi_margin_loss(logits, y_true)
        raise ValueError('unknown loss: ' + self.loss_name)

    def to_tensor(self, X):
        # pixel values are 0-255; scaling them to [0, 1] keeps the gradients well behaved
        return torch.FloatTensor(np.array(X) / 255.0).to(self.device)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y, test_X=None, test_y=None):
        self.to(self.device)
        optimizer = self.get_optimizer()

        X = self.to_tensor(X)
        y = torch.LongTensor(np.array(y)).to(self.device)
        n = X.shape[0]

        for epoch in range(self.max_epoch):
            # nn.Module.train() switches dropout on; we call it through nn.Module because
            # this class overrides train() with the template's training loop
            nn.Module.train(self, True)
            # shuffle the training set each epoch, then walk through it batch by batch
            permutation = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            correct = 0
            for start in range(0, n, self.batch_size):
                idx = permutation[start:start + self.batch_size]
                logits = self.forward(X[idx])
                train_loss = self.compute_loss(logits, y[idx])

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # update the variables according to the optimizer and the gradients calculated above
                optimizer.step()

                epoch_loss += train_loss.item() * len(idx)
                correct += (logits.argmax(1) == y[idx]).sum().item()

            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(epoch_loss / n)
            self.history['train_accuracy'].append(correct / n)
            # test accuracy is only recorded to draw the convergence curve; it is never used to pick the model
            if test_X is not None:
                self.history['test_accuracy'].append(float(np.mean(self.test(test_X) == np.array(test_y))))
            print('Epoch:', epoch + 1, 'Loss: %.4f' % self.history['train_loss'][-1],
                  'Train Accuracy: %.4f' % self.history['train_accuracy'][-1],
                  ('Test Accuracy: %.4f' % self.history['test_accuracy'][-1]) if test_X is not None else '')

    def test(self, X):
        # eval mode turns dropout off (self.eval() would call our overridden train(), so go through nn.Module);
        # no_grad skips building the autograd graph
        nn.Module.train(self, False)
        with torch.no_grad():
            y_pred = self.forward(self.to_tensor(X))
        # instances get the label with the largest score
        return y_pred.argmax(1).cpu().numpy()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'],
                   self.data['test']['X'], self.data['test']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'history': self.history}
