import json
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from .vocab import Labels, Vocab
from ..utils.ops import stack_with_padding


class SeqDataset(Dataset):
    """
    Format a sequence input for Encoder from text, including tokenization.
    """
    def __init__(self,
                 data_path: str,
                 tokenizer: AutoTokenizer,
                 vocab: Vocab = None,
                 label_path: str = None):

        with open(data_path, 'r') as f:
            self.data = json.load(f)
        if label_path:
            with open(label_path, 'r') as f:
                self.labels = json.load(f)

        # init labels and group by doc and clusters
        self.labels = Labels(self.labels)

        # unifying data, adding cluster id info
        self.data = self.labels.unify_corpus(self.data)

        # each sample is a sentence, each sample contains all tokens in a list
        # where each token is in a format of a tuple (doc_name, s_id, t_id)
        # TODO: Now dealing with sent individually, in future, considering of whole document info
        self.idx_to_sample = []
        sent = []
        for doc_name, doc_body in self.data.items():
            local_s_id = 0
            for token in doc_body:
                if token[0] != local_s_id:
                    self.idx_to_sample.append(sent)
                    sent = []
                    local_s_id += 1
                t_id = token[1] - 1
                sent.append((token[0], t_id, token[2], doc_name, token[-1]))

        # get entities vocab
        if vocab is None:
            self.vocab = Vocab()
            self.vocab.build(self.data, self.labels)
        else:
            self.vocab = vocab

        # spanBert related
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.idx_to_sample)

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        """
        Loads and returns a sample given index. Returns a dict with training input and target.
        Used for training.
        """
        sent = self.idx_to_sample[index]
        # tokenize inputs using spanBert
        inputs = [self.tokenizer.encode(token[2], add_special_tokens=True)[1] for token in sent]
        # getting targets
        targets_str = [self.labels.get_name_by_token(token) for token in sent]
        targets = self.vocab.vectorize(targets_str)
        return torch.tensor(inputs), torch.tensor(targets)

    def batch_fn(self, samples: List, device: torch.device) -> Tuple[Dict, Dict]:
        """
        A function for batching samples with paddings.
        Return:
            list of tensors for inputs and targets for training.
        """
        xs, ys = zip(*samples)

        # extract original lengths of each sample
        xs_lens = torch.tensor([len(x) for x in xs]).to(device)
        ys_lens = torch.tensor([len(y) for y in ys]).to(device)

        # pad and stack
        inputs = {
            "sentences": stack_with_padding(xs).to(device),
            "num_tokens": xs_lens
        }
        targets = {
            "labels": stack_with_padding(ys).to(device),
            "num_tokens": ys_lens
        }

        return inputs, targets


def fetch_dataloader(dataset: SeqDataset,
                     split: str,
                     batch_size: int,
                     device: torch.device,
                     num_workers: int = 0) -> torch.utils.data.DataLoader:
    """
    Get the dataloader accordingly with specific split.
    Args:
        dataset: the SeqDataset that contains data samples
        split: the string indicates which partition of data is required
        batch_size: the integer that wraps batch of samples
        num_workers: number of workers for GPU
    Returns:
        torch.utils.data.Dataloader: the torch Dataloader that is used for model
    """
    if split == "train":
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=lambda samples: dataset.batch_fn(samples, device),
                                           num_workers=num_workers)
    if split in ["val", "test"]:
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           collate_fn=lambda samples: dataset.batch_fn(samples, device),
                                           num_workers=0)







