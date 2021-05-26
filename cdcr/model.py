import torch
import torch.nn as nn

from transformers import AutoModel

from encoder import Encoder
from .encoder.independent_encoder import IndependentEncoder
from decoder import Decoder
from .decoder.rnn_decoder import RNNDecoder


class CDCRModel(nn.Module):
    """
    Model class that integrate encoder and decoder together. Taking tokenized inputs and predict relations between them.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, outputs=None):
        enc_output = self.encoder(inputs)
        relation_scores = self.decoder(enc_output)
        return relation_scores





def build_model(encoder_name: str,
                decoder_name: str,
                hidden_size: int,
                input_size: int,
                vocab_size: int,
                pre_trained_emb: AutoModel = None,
                ):
    # create encoder
    if encoder_name == 'independent':
        encoder = IndependentEncoder(pre_trained_emb=pre_trained_emb)

    # create decoder
    if decoder_name == "rnn":
        decoder = RNNDecoder(hidden_size=hidden_size,
                             input_size=input_size,
                             vocab_size=vocab_size)

    return CDCRModel(encoder, decoder)

