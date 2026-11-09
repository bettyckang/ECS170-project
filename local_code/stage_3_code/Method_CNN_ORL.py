'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_ORL(Method_CNN):
    # only 360 training faces (9 per person), so many short epochs with small batches
    max_epoch = 40
    batch_size = 32

    # input: 1 channel (the R channel of the gray-scale image), 112x92; output: 40 people
    # three conv blocks shrink 112x92 -> 56x46 -> 28x23 -> 14x11 before the classifier;
    # heavier dropout because the training set is tiny and easy to memorize
    def __init__(self, mName, mDescription, conv_channels=(16, 32, 64), kernel_size=3, padding=1, stride=1,
                 pool='max', hidden_dim=256, dropout=0.5, activation='relu', batch_norm=False):
        super().__init__(mName, mDescription, in_channels=1, image_size=(112, 92), num_classes=40,
                         conv_channels=conv_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                         pool=pool, hidden_dim=hidden_dim, dropout=dropout, activation=activation,
                         batch_norm=batch_norm)
