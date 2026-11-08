'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_CIFAR(Method_CNN):
    # 50,000 colored objects; the hardest of the three datasets, so more epochs
    max_epoch = 20
    batch_size = 128
    # with a fixed learning rate, CIFAR test accuracy swung 3-5 points between epochs;
    # lowering the rate over training lets the final epoch settle
    lr_schedule = 'cosine'

    # input: 3 color channels, 32x32; output: 10 object classes
    # three conv blocks (32x32 -> 16x16 -> 8x8 -> 4x4) with batch norm, which stabilizes training on color images
    def __init__(self, mName, mDescription, conv_channels=(32, 64, 128), kernel_size=3, padding=1, stride=1,
                 pool='max', hidden_dim=256, dropout=0.5, activation='relu', batch_norm=True):
        super().__init__(mName, mDescription, in_channels=3, image_size=(32, 32), num_classes=10,
                         conv_channels=conv_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                         pool=pool, hidden_dim=hidden_dim, dropout=dropout, activation=activation,
                         batch_norm=batch_norm)
