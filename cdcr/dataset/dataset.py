import json
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from .vocab import Labels, Vocab
from ..utils.ops import stack_with_padding


class SeqDataset(Dataset):
    """
    Format a sequence input for Encoder from text, including tokenization, sampling.
    Current sampling method is over-sampling.
    """
    def __init__(self,
                 data_path: str,
                 tokenizer: AutoTokenizer,
                 vocab: Vocab = None,
                 label_path: str = None,
                 sampling: str = None):

        # spanBert related
        self.tokenizer = tokenizer
        # action ids
        self.action_copy = 1
        self.action_not_copy = 0
        # copy id for targets
        self.copy_id = 0

        self.idx_to_sample = []
        # check if it's ecb/ontoNotes
        if 'ecb' in data_path:
            self.data_name = 'ecb'
            with open(data_path, 'r') as f:
                self.data = json.load(f)
            if label_path:
                with open(label_path, 'r') as f:
                    self.labels = json.load(f)

            self.vocab = vocab
            # self.entities_vocab = self.vocab.entities_dict

            # init labels and group by doc and clusters
            self.labels = Labels(self.labels)

            # unifying data, adding cluster id info
            self.data = self.labels.unify_corpus(self.data)
            # each sample is a sentence, each sample contains all tokens in a list
            # where each token is in a format of a tuple (token_str, t_id, s_id, doc_name, (cluster_id, span_len))
            # TODO: Now dealing with sent individually, in future, considering of whole document info
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

        elif 'ontoNotes' in data_path:
            self.data_name = 'ontoNotes'
            # getting raw corpus
            with open(data_path, 'r') as f:
                self.data = [json.loads(jsonline) for jsonline in f.readlines()]
            # each sample is one document segments with different sentences, maximum 512 items per doc
            dummy_cluster = 0
            for document in self.data:
                doc_name = document["doc_key"]
                t_id = 0
                doc_example = []
                for sentence in document['sentences']:
                    example = []
                    for t_str in sentence:
                        s_id = document['sentence_map'][t_id]
                        example.append([t_str, t_id, s_id, doc_name, dummy_cluster])
                        t_id += 1
                    doc_example.append(example)
                # assigning cluster id
                d_last = [e[-1][1] for e in doc_example]
                for c_id, cluster in enumerate(document['clusters']):
                    for span in cluster:
                        span_len = (span[1] - span[0] + 1)
                        for idx in range(span[0], span[-1] + 1):
                            for i, last in enumerate(d_last):
                                if idx <= last:
                                    e_id = i
                                    t_id = idx - d_last[i - 1] - 1 if i > 0 else idx
                                    break
                            doc_example[e_id][t_id][-1] = (c_id + 1, span_len)
                for e in doc_example:
                    self.idx_to_sample.append(e)

        # sampling
        if sampling:
            self.idx_to_sample = self.__sampling(sampling_method=sampling, idx_to_sample=self.idx_to_sample)

    def __sampling(self, sampling_method, idx_to_sample: List, num_samples: int=10) -> List:
        """
        Sampling method for training data: currently supports over-sampling.
        Over-sampling: simply sample data that contains labels.
        TODO: supports sampling between classes(entities)
        :param sampling_method:
        :param idx_to_sample:
        :return:
        """
        if sampling_method == "over-sampling":
            samples = []
            for sample in idx_to_sample:
                samples.append(sample)
                for token in sample:
                    if token[-1] != 0:
                        for _ in range(num_samples - 1):
                            samples.append(sample)
                        break
        return samples

    def __len__(self) -> int:
        return len(self.idx_to_sample)

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        """
        Loads and returns a sample given index. Returns a dict with training input and target.
        Used for training.
        """
        sent = self.idx_to_sample[index]
        # tokenize inputs using spanBert
        inputs = self.tokenizer.convert_tokens_to_ids([t[0] for t in sent])
        if self.data_name == 'ecb':
            if not self.entities_vocab:
                # getting targets
                targets_str = [self.labels.get_name_by_token(token) for token in sent]
            else:
                targets_str = [self.labels.get_copy_name_by_token(token) for token in sent]
            targets = self.vocab.vectorize(targets_str)
            copy_id = self.vocab["<copy>"]
            actions = [1 if t == copy_id else 0 for t in targets]
        elif self.data_name == "ontoNotes":
            # generate action sequence and closest antecedents ids
            # create a dictionary to store local {cluster_id: (antecedent_span_begin_token_id, antecedent_span_end_token_id)}
            actions = []
            targets = []
            cluster_dict = {}
            t_id, offset = sent[0][1], sent[0][1]
            while t_id <= sent[-1][1]:
                token = sent[t_id - offset]
                if token[-1] != 0:
                    c_id, span_len = token[-1]
                    # put antecedents in the targets
                    if c_id in cluster_dict:
                        antecedent_ids = cluster_dict[c_id]
                        # adding end of input span, start and end of antecedent span
                        targets.append(t_id - offset + span_len - 1)
                        targets.append(antecedent_ids[0] - offset)
                        targets.append(antecedent_ids[1] - offset)
                        t_id += span_len
                        # TODO: remove hardcoded number
                        # only add three not copy actions for end, start, end
                        for _ in range(3):
                            actions.append(self.action_not_copy)
                    else:
                        for _ in range(span_len):
                            targets.append(t_id - offset)
                            actions.append(self.action_copy)
                            t_id += 1
                    # update cluster antecedent
                    cluster_dict[c_id] = (token[1], token[1] + span_len - 1)
                else:
                    actions.append(self.action_copy)
                    targets.append(t_id - offset)
                    t_id += 1

        return torch.tensor(inputs), torch.tensor(targets), torch.tensor(actions).float()

    def batch_fn(self, samples: List, device: torch.device) -> Tuple[Dict, Dict]:
        """
        A function for batching samples with paddings.
        Return:
            list of tensors for inputs and targets for training.
        """
        xs, ys, zs = zip(*samples)

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
            "actions": stack_with_padding(zs).to(device),
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







