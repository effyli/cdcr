import json
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from .corpus import Labels
from ..utils.vocab import EntVocab
from ..utils.ops import stack_with_padding


class SeqDataset(Dataset):
    """
    Format a sequence input for Encoder from text, including tokenization.
    """
    def __init__(self,
                 data_path: str,
                 tokenizer: AutoTokenizer,
                 entities_vocab: EntVocab = None,
                 label_path: str = None):

        with open(data_path, 'r') as f:
            self.data = json.load(f)
        if label_path:
            with open(label_path, 'r') as f:
                self.labels = json.load(f)

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
                sent.append((doc_name, token[0], t_id, token[2]))

        # get entities vocab
        if entities_vocab is None:
            self.entVocab = EntVocab()
            self.entVocab.build(self.labels)
        else:
            self.entVocab = entities_vocab

        # init labels and group by doc
        self.labels = Labels(self.labels)

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
        inputs = []
        for token in sent:
            inputs.append(self.tokenizer.encode(token[-1], add_special_tokens=True)[1])

        labels_by_doc = self.labels.labels_by_doc
        # getting labels for the sentence in a list of Mentions objects
        target_mentions = []
        for (d_name, s_id, t_id, _) in sent:
            # in sent, token id starts from 0
            t_mention = None
            if (d_name in labels_by_doc) and (str(s_id) in labels_by_doc[d_name]):
                for mention in labels_by_doc[d_name][str(s_id)]:
                    if t_id in mention.tokens_ids:
                        t_mention = mention
                        break
            if t_mention:
                target_mentions.append(t_mention)
            else:
                target_mentions.append(self.entVocab.unk)

        # vectorize target mentions
        targets = self.entVocab.vectorize(target_mentions)

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







