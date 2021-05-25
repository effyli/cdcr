from typing import Iterable, List, Any
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


class EntVocab:
    """
    An entity vocabulary to deal with vectorization of labels
    TODO: be able to define the vocab size as a hyperparameter
    """

    def __init__(self):
        """
        Initialization of the vocabulary object
        All labels that are not belong to any entities are identified as unk
        dictionary contains ent_id: entity.
        m_dictionary contains entity_mentions: entity
        reversed_dictionary contains entity: ent_id
        """
        # create special entity for unk
        # TODO: we also need pad
        unk = {}
        self.unk = Mention(**unk)
        self.__special_symbols = [self.unk]
        # stores entity: e_id
        self.__dictionary = {}
        # stores e_id: entity
        self.__reversed_dictionary = {}
        # stores cluster_id: entity
        self.__clusters = {}


    @property
    def size(self):
        return len(self.__dictionary) + 1

    def __getitem__(self, item):
        return

    def build(self, labels: Iterable[Iterable[Any]]) -> None:
        """
        Build the entity vocabulary based on the labels
        """
        for symbol in self.__special_symbols:
            self.__dictionary[symbol] = len(self.__dictionary)
        for label in labels:
            mention = Mention(**label)
            c_id = mention.cluster_id
            if c_id not in self.__clusters:
                ent = Entity(c_id)
                ent.add_mention(mention)
                self.__dictionary[ent] = len(self.__dictionary)
                self.__clusters[c_id] = ent
            else:
                ent = self.__clusters[c_id]
                ent.add_mention(mention)
        self.__reversed_dictionary = {val: key for (key, val) in self.__dictionary.items()}

    def vectorize(self, labels: Iterable[Any]) -> List:
        """
        Vectorize a sequence of labels to ids
        """
        vector = []
        for label in labels:
            mention = label if isinstance(label, Mention) else Mention(**label)
            c_id = mention.cluster_id
            if c_id in self.__clusters:
                vector.append(self.__dictionary[self.__clusters[c_id]])
            else:
                vector.append(self.__dictionary[self.unk])
        return vector

    def devectorize(self, vector: Iterable[int]) -> List[Any]:
        """
        Devectorize a list of ids into corresponding entities
        """
        entities = []
        for idx in vector:
            if idx > 0:
                entities.append(self.__reversed_dictionary[idx])
            else:
                entities.append(self.unk)
        return entities

