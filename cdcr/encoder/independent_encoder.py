import torch
import torch.nn as nn

from transformers import AutoModel
from .encoder import Encoder


class IndependentEncoder(Encoder):
    """
    An encoder holds independent assumption (without context) for each token in the input.
    """

    def __init__(self, pre_trained_emb: AutoModel = None):
        super().__init__()
        # TODO: adding attention?
        self.pre_trained_emb = pre_trained_emb

    def forward(self, inputs):
        # using pre-trained bert model, last layer hidden states
        token_embeddings = self.pre_trained_emb(inputs["sentences"])[0]

        # mask padding
        masks = torch.zeros_like(inputs["sentences"], dtype=torch.float)
        for row, col in zip(masks, inputs["num_tokens"]):
            row[:col] = 1
        token_embeddings = token_embeddings * torch.unsqueeze(masks, 2)

        return token_embeddings


