from abc import ABCMeta
import sys

import torch.nn as nn


class Decoder(nn.Module, metaclass=ABCMeta):
    """
    Abstract class for decoder. Give outputs from encoder and predicts relations between tokens.
    """
    def __init__(self, *params):
        self.params = params
        super().__init__()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dict['class_name'] = self.__class__.__name__
        state_dict['params'] = self.params
        return state_dict


def load_decoder(class_name: str, params):
    class_ = getattr(sys.modules["cdcr.decoder"], class_name)
    return class_(*params)
