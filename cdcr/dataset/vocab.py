import pickle

from collections import defaultdict
from typing import Iterable, List, Any, Dict, Union
from dataclasses import dataclass

@dataclass(eq=True, frozen=True)
class Mention:
    """
    A mention dataclass to deal with each label
    """
    doc_id: str = ''
    subtopic: str = ''
    topic: str = ''
    m_id: str = ''
    sentence_id: str = ''
    tokens_ids: list = ''
    tokens: str = ''
    tags: str = ''
    lemmas: str = ''
    cluster_id: int = 0
    cluster_desc: str = ''
    singleton: bool = False


class Entity:
    """
    An entity class
    """
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.mentions = []

    def add_mention(self, mention):
        self.mentions.append(mention)

    def __str__(self):
        return "{}_{}".format(self.cluster_id, self.mentions[0].cluster_desc)



class Labels:
    """
    A label class for storing useful information for all labels. Use for dataset
    - grouping labels by documents
    """
    def __init__(self, labels: List):
        self.unk_id = 0
        self.labels = labels
        # {doc_name: {sent_id: [mentions]}}
        self.labels_by_doc = {}
        # {cluster_id: entity}
        self.__clusters = {}
        self.group_labels()

    def group_labels(self):
        """
        A function to group all labels by docs and sentences.
        """
        for label in self.labels:
            label = Mention(**label)
            # group current label
            s_id = label.sentence_id
            d_name = label.doc_id
            if d_name not in self.labels_by_doc:
                self.labels_by_doc[d_name] = {s_id: [label]}
            elif s_id not in self.labels_by_doc[d_name]:
                self.labels_by_doc[d_name][s_id] = [label]
            else:
                self.labels_by_doc[d_name][s_id].append(label)
            # cluster in corresponding entity
            c_id = label.cluster_id
            if c_id not in self.__clusters:
                ent = Entity(c_id)
                ent.add_mention(label)
                self.__clusters[c_id] = ent
            else:
                ent = self.__clusters[c_id]
                ent.add_mention(label)

    def get_clusters(self):
        return self.__clusters

    def unify_corpus(self, corpus: Dict):
        """
        Unifying a token string to it corresponding entity string if available
        :param corpus: a dictionary contains all data in a format of {doc_name: tokens}
        :return: a dictionary of keys for doc name and tokens with corresponding cluster id in the end
        """
        unified = {}
        for doc_name, tokens in corpus.items():
            labels = self.labels_by_doc[doc_name]
            unified_tokens = []
            for token in tokens:
                cluster_id = self.unk_id
                s_id, t_id, token_str = token[:3]
                if str(s_id) in labels:
                    for mention in labels[str(s_id)]:
                        if t_id in mention.tokens_ids:
                            cluster_id = mention.cluster_id
                token.append(cluster_id)
                unified_tokens.append(token)
            unified[doc_name] = unified_tokens
        return unified

    def get_name_by_token(self, token):
        return str(self.__clusters[token[4]]) if token[4] in self.__clusters else token[2]


class Vocab:
    """
    A vocabulary class for decoder to generate tokens from.
    """

    def __init__(self, min_frequency: int = 2, min_sentence_freq: int = 1):
        self.__special_symbols = ["<unk>", "<pad>", "<sos>", "<eos>"]

        self.__min_frequency = min_frequency
        self.__min_sentence_freq = min_sentence_freq
        self.__dictionary = {}
        self.__reversed_dictionary = {}

        # entity_name: entity
        self.__entities_dict = {}

    def build(self, corpus: Dict, labels: Union[Labels, List]):
        """
        Build vocabulary from given corpus
        :param corpus: a unified corpus with cluster info
        :param labels:
        :return:
        """
        labels = labels if isinstance(labels, Labels) else Labels(labels)
        all_counts = defaultdict(lambda: 0)
        self.__entities_dict = {str(ent): ent for _, ent in labels.get_clusters().items()}

        for special_symbol in self.__special_symbols:
            self.__dictionary[special_symbol] = len(self.__dictionary)

        for doc_name, tokens in corpus.items():
            for token in tokens:
                token_str = labels.get_name_by_token(token)
                all_counts[token_str] += 1
                if token_str not in self.__dictionary and \
                        (all_counts[token_str] >= self.__min_frequency or token_str in self.__entities_dict):
                    self.__dictionary[token_str] = len(self.__dictionary)

        self.__reversed_dictionary = {val: key for (key, val) in self.__dictionary.items()}

    @property
    def size(self):
        return len(self.__dictionary) + 1

    def __getitem__(self, token):
        token_id = self.__dictionary.get(token)
        if token_id is not None:
            return token_id
        return len(self.__dictionary)

    def vectorize(self, sequence: Iterable[Any]):
        """
        Vectorize the sequence based on the vocab
        """
        return [self.__dictionary[token] if token in self.__dictionary
                else self.__dictionary["<unk>"] for token in sequence]

    def devectiorize(self, vector: Iterable[int]):
        """
        Devectorizes a list of ids into their corresponding tokens

        Args:
            vector (Iterable[int]): The list of ids to convert back to tokens

        Returns:
            List[Any]: A list of the corresponding tokens
        """
        return [self.__reversed_dictionary[index] if index > 0 else 'UNK' for index in vector]


def build_vocab(config):
    # optional: build vocabulary across dataset
    import json
    vocab = Vocab()
    label_path = "../data/ecb/mentions/train_entities.json"
    val_label_path = "../data/ecb/mentions/dev_entities.json"
    test_label_path = "../data/ecb/mentions/test_entities.json"
    with open(label_path, 'r') as f:
        labels = json.load(f)
    with open(val_label_path, 'r') as f:
        labels.extend(json.load(f))
    with open(test_label_path, 'r') as f:
        labels.extend(json.load(f))

    data_path = "../data/ecb/mentions/train.json"
    val_data_path = "../data/ecb/mentions/dev.json"
    test_data_path = "../data/ecb/mentions/test.json"
    data_paths = [data_path, val_data_path, test_data_path]
    data = {}
    for path in data_paths:
        with open(path, 'r') as f:
            data.update(json.load(f))

    labels = Labels(labels)
    data = labels.unify_corpus(data)

    vocab.build(data, labels)

    with open(config.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
