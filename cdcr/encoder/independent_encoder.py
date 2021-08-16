import torch
import torch.nn as nn

from transformers import AutoModel
from .encoder import Encoder


class IndependentEncoder(Encoder):
    """
    An encoder holds independent assumption (without context) for each token in the input.
    """

    def __init__(self, bert_size: int, pre_trained_emb: AutoModel = None):
        super().__init__(bert_size, pre_trained_emb)
        self.pre_trained_emb = pre_trained_emb
        self.rnn = nn.LSTM(bert_size, bert_size)

    def forward(self, inputs):
        # using pre-trained bert model, last layer hidden states
        outputs = self.pre_trained_emb(inputs["sentences"])
        token_embeddings, pooler_output = outputs[0], outputs[1]
        token_embeddings, _ = self.rnn(token_embeddings)

        # mask padding
        masks = torch.zeros_like(inputs["sentences"], dtype=torch.float)
        for row, col in zip(masks, inputs["num_tokens"]):
            row[:col] = 1
        token_embeddings = token_embeddings * torch.unsqueeze(masks, 2)
        # sum_token_embeddings = token_embeddings.sum(1)
        # sum_token_embeddings /= torch.unsqueeze(inputs["num_tokens"], 1)

        return token_embeddings, pooler_output


