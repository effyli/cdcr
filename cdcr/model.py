from collections import defaultdict

import torch
import torch.nn as nn

from transformers import AutoModel

from .encoder.encoder import Encoder, load_encoder
from .encoder.independent_encoder import IndependentEncoder
from .decoder import Decoder, load_decoder, RNNDecoder, CopyDecoder


class CDCRModel(nn.Module):
    """
    Model class that integrate encoder and decoder together. Taking tokenized inputs and predict relations between them.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, outputs=None):
        hidden_states = self.encoder(inputs)
        relation_scores = self.decoder(hidden_states, outputs)
        return relation_scores

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = defaultdict()
        state['encoder_state'] = self.encoder.state_dict(destination, prefix, keep_vars)
        state['decoder_state'] = self.decoder.state_dict(destination, prefix, keep_vars)
        return state

    def load_state_dict(self, state_dict, strict=True):
        del state_dict['encoder_state']['class_name']
        del state_dict['decoder_state']["class_name"]
        del state_dict['encoder_state']["params"]
        del state_dict['decoder_state']["params"]

        self.encoder.load_state_dict(state_dict["encoder_state"])
        self.decoder.load_state_dict(state_dict['decoder_state'])

    def save(self, path: str):
        state = self.state_dict()
        torch.save(state, path)

    def load(self, path: str = None, device: torch.device = 'cpu', state=None):
        state = torch.load(path, map_location=device)
        self.load_state_dict(state)

    @staticmethod
    def loads(path: str, device: torch.device):
        state = torch.load(path, map_location=device)
        topic_model = load_encoder(
            state['encoder_state']['class_name'],
            state['encoder_state']['params'])
        behaviour_predictor = load_decoder(
            state['decoder_state']['class_name'],
            state['decoder_state']['params'])
        model = CDCRModel(topic_model, behaviour_predictor)
        model.load_state_dict(state)
        return model


def build_model(encoder_name: str,
                decoder_name: str,
                hidden_size: int,
                input_size: int,
                vocab_size: int,
                sos_id: int,
                eos_id: int,
                copy_id: int = None,
                pre_trained_emb: AutoModel = None,
                ):
    # create encoder
    if encoder_name == 'independent':
        encoder = IndependentEncoder(pre_trained_emb=pre_trained_emb)

    # create decoder
    if decoder_name == "rnn":
        decoder = RNNDecoder(hidden_size=hidden_size,
                             input_size=input_size,
                             vocab_size=vocab_size,
                             sos_id=sos_id,
                             eos_id=eos_id)
    elif decoder_name == "copy":
        decoder = CopyDecoder(
            hidden_size=hidden_size,
            input_size=input_size,
            vocab_size=vocab_size,
            sos_id=sos_id,
            eos_id=eos_id,
            copy_id=copy_id
        )

    return CDCRModel(encoder, decoder)

