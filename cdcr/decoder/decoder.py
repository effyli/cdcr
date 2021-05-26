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


