from abc import ABCMeta

import torch.nn as nn


class Encoder(nn.Module, metaclass=ABCMeta):
    """
    Abstract class for Encoder in the framework that encodes each token in the input, and providing
    useful linguistic information.
    """
    def __init__(self, *params):
        self.params = params
        super(Encoder, self).__init__()

