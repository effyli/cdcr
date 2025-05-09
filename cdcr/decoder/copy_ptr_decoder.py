from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from .decoder import Decoder
from ..utils.model_utils import bernoulli_action_log_prob


# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_max(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int,
                   keepdim: bool = False,
                   min_val: float = -1e7) -> (torch.Tensor, torch.Tensor):
    """
    To calculate max along certain dimensions on masked values
    Parameters
    ----------
    vector : ``torch.Tensor``
        The vector to calculate max, assume unmasked parts are already zeros
    mask : ``torch.Tensor``
        The mask of the vector. It must be broadcastable with vector.
    dim : ``int``
        The dimension to calculate max
    keepdim : ``bool``
        Whether to keep dimension
    min_val : ``float``
        The minimal value for paddings
    Returns
    -------
    A ``torch.Tensor`` of including the maximum values.
    """
    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value, max_index


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1 (unsqueezed), hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)

        # 1st line of Eq.(3) in the paper
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

        # softmax with only valid inputs, excluding zero padded parts
        # log-softmax for a better numerical stability
        log_score = masked_log_softmax(u_i, mask, dim=-1)

        return log_score


class CopyPtrDecoder(Decoder):
    """
    A lstm decoder combined with copy + pointer network
    """

    def __init__(self,
                 hidden_size: int,
                 sos_id: int,
                 eos_id: int,
                 copy_id: int,
                 input_size: int,
                 pre_trained_emb: AutoModel,
                 max_iterations: int=200,
                 bidirectional: bool = False):
        """
        Args:
            hidden_size: the hidden dimension for LSTM
            input_size: the dimension from the output of encoder
        """
        super().__init__(hidden_size, sos_id, eos_id, copy_id, input_size, pre_trained_emb, max_iterations, bidirectional)
        self.bidirectional = bidirectional
        self.max_iterations = max_iterations

        # decoding rnn
        self.rnn = nn.LSTMCell(hidden_size, hidden_size)
        self.embedding = pre_trained_emb

        # decision making for copy
        self.copy = 1
        self.decision_making = nn.Linear(hidden_size * (2 if bidirectional else 1), 1, bias=False)

        self.input_end_attn = Attention(hidden_size)
        self.ant_start_attn = Attention(hidden_size)
        self.ant_end_attn = Attention(hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        if input_size != hidden_size:
            self.initial_transform = nn.Linear(input_size, hidden_size, bias=False)
            self.initial_input_transform = nn.Linear(input_size, hidden_size, bias=False)
        else:
            self.initial_transform = None
            self.initial_input_transform = None

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.copy_id = copy_id

    def forward(self, encoder_outputs, initial_state, inputs, targets):
        """
        Estimate the distribution over vocabulary at each timestep
        Shapes:
        Targets: [B, S]
        Args:
            encoder_outputs: [B, S, H], an output embedding from Bert for each token
            initial_state: average sum of all tokens
            inputs: vectorized inputs from encoder
            targets: expected relation class for each step
        Return:
            Scores over vocabulary
        """
        # TODO: getting attentive initial states.
        if self.initial_transform is not None:
            initial_state = self.initial_transform(initial_state)

        batch_size = initial_state.size(0)
        max_input_len = inputs['sentences'].size(1)
        max_output_len = max(targets['num_tokens'])
        device = initial_state.device

        actions = targets['actions']
        labels = targets['labels']
        # create <sos> token as the first time step input
        initial_input = torch.tensor([self.sos_id for _ in range(batch_size)]).unsqueeze(1).to(device)
        rnn_input = self.embedding(initial_input)[0].squeeze(1)
        if self.initial_input_transform is not None:
            rnn_input = self.initial_input_transform(rnn_input)
        rnn_hidden = initial_state

        if self.bidirectional:
            rnn_hidden = torch.stack([rnn_hidden, rnn_hidden], 0)
        rnn_hidden = (rnn_hidden, rnn_hidden)

        # used for inference
        batch_decisions = torch.tensor([self.copy for _ in range(batch_size)]).to(device)
        batch_decisions = batch_decisions.unsqueeze(0).unsqueeze(-1)

        # create mask tensors
        range_tensor = torch.arange(max_input_len, device=device).expand(
            batch_size, max_input_len, max_input_len)
        each_len_tensor = inputs['num_tokens'].view(-1, 1, 1).expand(batch_size, max_input_len, max_input_len)

        row_mask_tensor = (range_tensor < each_len_tensor)
        col_mask_tensor = row_mask_tensor.transpose(1, 2)
        mask_tensor = row_mask_tensor * col_mask_tensor

        log_outputs = []
        outputs_ids = []
        action_probs = []
        end_input_log_pointer_scores = []
        end_input_pointer_argmaxs = []
        start_ant_log_pointer_scores = []
        start_ant_pointer_argmaxs = []
        end_ant_log_pointer_scores = []
        end_ant_pointer_argmaxs = []

        pointing_starts = [False for _ in range(batch_size)]
        pointing_ends = [False for _ in range(batch_size)]
        batch_input_index = [self.sos_id for _ in range(batch_size)]

        is_finished = [False for _ in range(batch_size)]

        # for each time step we first make a decision of copy or not
        for step in range(max_output_len):
            rnn_hidden = self.rnn(rnn_input, rnn_hidden)
            h_step, c_step = rnn_hidden
            h_step = torch.relu(h_step)
            rnn_hidden = h_step, c_step
            # get the logit, used for calculate the log prob
            logit = self.decision_making(h_step)
            logit.retain_grad()
            # this prob is used for inference
            prob = torch.sigmoid(logit)
            batch_actions = actions[:, step]
            batch_labels = labels[:, step]

            batch_action_probs = []
            batch_log_output = []
            batch_id_output = []

            batch_rnn_input = []

            # unroll the batch
            for i in range(batch_size):
                # get the log prob
                action = batch_actions[i]
                pointing_start = pointing_starts[i]
                pointing_end = pointing_ends[i]

                batch_action_probs.append(bernoulli_action_log_prob(logit[i], action).to(device))
                # print(batch_input_index)
                sub_mask = mask_tensor[:, batch_input_index[i], :].float()
                sub_mask_i = sub_mask[i].unsqueeze(0)
                # during training, we use ground truth actions
                # if not copy, use a pointer net to point to specific token until next decision is copy
                # unroll the batch. within one batch, there are copy and not copy
                if not pointing_start and not pointing_end and not action:
                    # predict the end position of input span
                    # calculate pointer score for all encoder outputs
                    # predict the end input span position
                    end_input_log_pointer_score = self.input_end_attn(h_step[i].unsqueeze(0), encoder_outputs[i].unsqueeze(0), sub_mask_i)
                    end_input_log_pointer_scores.append(end_input_log_pointer_score)
                    # Get the indices of maximum pointer
                    _, masked_argmax = masked_max(end_input_log_pointer_score, sub_mask_i, dim=1, keepdim=True)
                    end_input_pointer_argmaxs.append(masked_argmax)
                    index_tensor = masked_argmax.expand(1, self.hidden_size)
                    batch_log_output.append(end_input_log_pointer_score.squeeze(0))
                    batch_id_output.append(masked_argmax.squeeze(0))
                    # feed in golden mentions instead of predicted index
                    batch_input_index[i] = int(batch_labels[i]) + 1 if int(batch_labels[i]) < max_input_len - 1 else int(batch_labels[i])
                    # index_tensor = batch_labels[i].unsqueeze(-1).expand(1, 1, self.hidden_size)
                    pointing_starts[i] = True
                    pointing_ends[i] = False

                elif pointing_start and not pointing_end and not action:
                    # predict the start position of antecedent span
                    start_ant_log_pointer_score = self.ant_start_attn(h_step[i].unsqueeze(0), encoder_outputs[i].unsqueeze(0), sub_mask_i)
                    start_ant_log_pointer_scores.append(start_ant_log_pointer_score)
                    _, masked_argmax = masked_max(start_ant_log_pointer_score, sub_mask_i, dim=1, keepdim=True)
                    start_ant_pointer_argmaxs.append(masked_argmax)
                    index_tensor = masked_argmax.expand(1, self.hidden_size)
                    batch_log_output.append(start_ant_log_pointer_score.squeeze(0))
                    batch_id_output.append(masked_argmax.squeeze(0))

                    pointing_ends[i] = True
                    pointing_starts[i] = False
                elif not pointing_start and pointing_end and not action:
                    # predict the end position of antecedent span
                    end_ant_log_pointer_score = self.ant_end_attn(h_step[i].unsqueeze(0), encoder_outputs[i].unsqueeze(0), sub_mask_i)
                    end_ant_log_pointer_scores.append(end_ant_log_pointer_score)
                    _, masked_argmax = masked_max(end_ant_log_pointer_score, sub_mask_i, dim=1, keepdim=True)
                    end_ant_pointer_argmaxs.append(masked_argmax)
                    index_tensor = masked_argmax.expand(1, self.hidden_size)
                    batch_log_output.append(end_ant_log_pointer_score.squeeze(0))
                    batch_id_output.append(masked_argmax.squeeze(0))
                    # next input index should be the end of input span + 1
                    # index_tensor = torch.tensor(batch_input_index[i]).unsqueeze(-1).exapnd(1,self.hidden_size)

                    pointing_ends[i] = False
                    pointing_starts[i] = False
                # if copy, generate current input id
                elif action:
                    pointer = batch_input_index[i]
                    index_tensor = torch.tensor(pointer).expand(1, self.hidden_size)
                    pointer_one_hot = torch.nn.functional.one_hot(torch.tensor([pointer, max_input_len - 1]))[0]
                    batch_log_output.append(pointer_one_hot)
                    batch_id_output.append(torch.tensor([pointer]))
                    # (batch_size, hidden_size)
                    pointing_ends[i] = False
                    pointing_starts[i] = False
                    batch_input_index[i] = pointer + 1 if pointer < max_input_len - 1 else pointer
                else:
                    print("category not meet")
                single_rnn_input = torch.gather(encoder_outputs[i], dim=0, index=index_tensor).squeeze(1)
                batch_rnn_input.append(single_rnn_input)
            action_probs.append(torch.tensor(batch_action_probs).to(device))
            log_outputs.append(torch.stack(batch_log_output))
            outputs_ids.append(torch.stack(batch_id_output))
            rnn_input = torch.stack(batch_rnn_input).squeeze(1)

        action_probs = torch.stack(action_probs).permute(1, 0)
        log_outputs = torch.stack(log_outputs).permute(1, 0, 2)
        outputs_ids = torch.stack(outputs_ids).permute(1,2,0)
        return {"action_probs": action_probs, "log_outputs": log_outputs.float(), "outputs_ids": outputs_ids}

