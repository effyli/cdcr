from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


from .decoder import Decoder


class RNNDecoder(Decoder):
    """
    Bi-LSTM decoder to decode relations from inputs.
    """
    
    def __init__(self,
                 hidden_size: int,
                 input_size: int,
                 vocab_size: int):
        """
        Args:
            hidden_size: the hidden dimension for LSTM
            input_size: the dimension from the output of encoder
            vocab_size: number of classes/entities possible to output at each timestep
        """
        super().__init__()

        self.biLSTM = nn.GRU(input_size, hidden_size, bidirectional=True)
        self.relation_output = nn.Linear(hidden_size * 2, vocab_size, bias=False)

    def forward(self, inputs):
        """
        Estimate the distribution over vocabulary at each timestep
        Args:
            inputs: vectorized inputs from encoder
            targets: expected relation class for each step
        Return:
            Scores over vocabulary
        """
        out, _ = self.biLSTM(inputs)
        relation_space = self.relation_output(out)
        relation_scores = F.log_softmax(relation_space, dim=1)

        return relation_scores
    

