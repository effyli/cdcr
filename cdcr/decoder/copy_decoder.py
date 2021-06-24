from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Decoder
from ..utils.model_utils import bernoulli_action_log_prob


class CopyDecoder(Decoder):
    """
    A lstm decoder combined with copy mechanism
    """

    def __init__(self,
                 hidden_size: int,
                 sos_id: int,
                 eos_id: int,
                 copy_id: int,
                 input_size: int,
                 vocab_size: int,
                 bidirectional: bool = False):
        """
        Args:
            hidden_size: the hidden dimension for LSTM
            input_size: the dimension from the output of encoder
            vocab_size: number of classes/entities possible to output at each timestep
        """
        super().__init__(hidden_size, input_size, sos_id, eos_id, vocab_size)
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(hidden_size, hidden_size)
        # should we use embedding for the output?
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # decision making for copy
        self.copy = 1
        self.decision_making = nn.Linear(hidden_size, 1)

        self.output_layer = nn.Linear(
            hidden_size * (2 if bidirectional else 1), vocab_size, bias=False)

        if input_size != hidden_size:
            self.initial_transform = nn.Linear(input_size, hidden_size, bias=False)
        else:
            self.initial_transform = None

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.copy_id = copy_id
        self.vocab_size = vocab_size

    def forward(self, initial_state, targets):
        """
        Estimate the distribution over vocabulary at each timestep
        Shapes:
        Targets: [B, S]
        Args:
            inputs: vectorized inputs from encoder
            targets: expected relation class for each step
        Return:
            Scores over vocabulary
        """
        # TODO: getting attentive initial states.
        if self.initial_transform is not None:
            initial_state = self.initial_transform(initial_state)

        num_steps = targets['labels'].size(1)
        batch_size = initial_state.size(0)
        device = initial_state.device

        actions = targets['actions']
        # create <sos> token as the first time step input
        initial_input = torch.tensor([self.sos_id for _ in range(batch_size)]).to(device)

        rnn_input = self.embedding(initial_input)
        rnn_hidden = initial_state
        outputs = []
        log_probs = []

        rnn_input = torch.unsqueeze(rnn_input, 0)
        rnn_hidden = torch.unsqueeze(rnn_hidden, 0)

        # used for inference
        batch_decisions = torch.tensor([self.copy for _ in range(batch_size)]).to(device)
        batch_decisions = batch_decisions.unsqueeze(0).unsqueeze(-1)

        # TODO: the masking?
        # for each time step we first make a decision of copy or not
        for step in range(num_steps):
            rnn_output, rnn_hidden = self.rnn(rnn_input, rnn_hidden)
            # get the logit, used for calculate the log prob
            logit = self.decision_making(rnn_output)
            # this prob is used for inference
            prob = torch.sigmoid(logit)
            batch_actions = actions[:, step]
            batch_log_probs = []
            batch_output = []
            # TODO: unroll the batch
            for i in range(batch_size):
                # get the log prob
                action = batch_actions[i]
                batch_log_probs.append(bernoulli_action_log_prob(logit[0][i], action))
                # during training, we use ground truth actions
                # if not copy, try to generate from output vocabulary conditioned on current generated results not on decision sequence
                # TODO: unroll the batch. within one batch, there are copy and not copy
                if not action:
                    output = self.output_layer(rnn_output[:, i, :].unsqueeze(0).permute(1, 0, 2))
                    # scores = F.log_softmax(output, dim=1)
                    # predicted_out = torch.argmax(scores, dim=2)
                    batch_output.append(output)
                # if copy, generate copy id
                # Question: will the next decision made on copy id?
                else:
                    idx = torch.tensor([4])
                    output = torch.zeros(len(idx), self.vocab_size).scatter_(1, idx.unsqueeze(1), 1.)
                    batch_output.append(output.unsqueeze(0).to(device))
            log_probs.append(torch.tensor(batch_log_probs))
            outputs.append(torch.stack(batch_output))

            rnn_input = self.embedding(targets["labels"][:, step])
            rnn_input = torch.unsqueeze(rnn_input, 0)

        log_probs = torch.stack(log_probs).permute(1, 0)
        outputs = torch.stack(outputs).squeeze(2).squeeze(2).permute(1, 0, 2)
        return {"log_probs": log_probs, "outputs": outputs}

