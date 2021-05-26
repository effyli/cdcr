import sys
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

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dict['class_name'] = self.__class__.__name__
        state_dict['params'] = self.params
        return state_dict


def load_encoder(class_name: str, params):
    class_ = getattr(sys.modules["cdcr.encoder"], class_name)
    return class_(*params)

