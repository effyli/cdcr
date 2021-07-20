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
                 sos_id: int,
                 eos_id: int,
                 input_size: int,
                 vocab_size: int,
                 bidirectional: bool = True):
        """
        Args:
            hidden_size: the hidden dimension for LSTM
            input_size: the dimension from the output of encoder
            vocab_size: number of classes/entities possible to output at each time step
        """
        super().__init__(hidden_size, input_size, sos_id, eos_id, vocab_size)
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(hidden_size, hidden_size, bidirectional=bidirectional)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.output_layer = nn.Linear(
            hidden_size * (2 if bidirectional else 1), vocab_size, bias=False)

        if input_size != hidden_size:
            self.initial_transform = nn.Linear(input_size, hidden_size, bias=False)
        else:
            self.initial_transform = None

        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, initial_state, targets):
        """
        Estimate the distribution over vocabulary at each timestep
        Args:
            inputs: vectorized inputs from encoder
            targets: expected relation class for each step
        Return:
            Scores over vocabulary
        """
        if self.initial_transform is not None:
            initial_state = self.initial_transform(initial_state)

        outputs = self.teacher_forcing(initial_state, targets)
        scores = F.log_softmax(outputs, dim=1)
        predicted_labels = torch.argmax(scores, dim=2)

        return scores

    def teacher_forcing(self, initial_state, inputs, targets):
        """
        Given the initial states from the encoder, predict the distribution over
        the output vocabulary for each time step with known input.

        :param initial_state: the topic vectors extracted the
        :param targets: the entity-ized symbol at each step to feed to the next step.
        :return:
        """
        batch_size = initial_state.size(0)

        # create <sos> token as the first time step input
        initial_input = torch.tensor([self.sos_id for _ in range(batch_size)]).to(initial_state.device)

        # and shift targets
        initial_input = torch.unsqueeze(initial_input, 1)
        rnn_inputs = torch.cat([initial_input, targets["labels"]], dim=1)

        rnn_inputs = self.embedding(rnn_inputs)
        # permute?
        rnn_inputs = rnn_inputs.permute(1, 0, 2)

        if self.bidirectional:
            initial_state = torch.stack([initial_state, initial_state], 0)
        else:
            initial_state = torch.unsqueeze(initial_state, 0)

        outputs, _ = self.rnn(rnn_inputs, initial_state)
        outputs = self.output_layer(outputs.permute(1, 0, 2))

        return outputs

