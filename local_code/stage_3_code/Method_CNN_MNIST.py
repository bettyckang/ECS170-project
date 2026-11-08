'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_MNIST(Method_CNN):
    # 60,000 small gray-scale digits: a few epochs are enough
    max_epoch = 10
    batch_size = 128

    # input: 1 channel, 28x28; output: 10 digits
    # two conv blocks: 28x28 -> 14x14 -> 7x7 feature maps, then a 128-unit hidden layer
    def __init__(self, mName, mDescription, conv_channels=(32, 64), kernel_size=3, padding=1, stride=1,
                 pool='max', hidden_dim=128, dropout=0.25, activation='relu', batch_norm=False):
        super().__init__(mName, mDescription, in_channels=1, image_size=(28, 28), num_classes=10,
                         conv_channels=conv_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                         pool=pool, hidden_dim=hidden_dim, dropout=dropout, activation=activation,
                         batch_norm=batch_norm)
