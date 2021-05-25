from typing import List
from ..utils.vocab import Mention


class Labels:
    """
    A label class for storing useful information for all labels. Use for dataset
    - grouping labels by documents
    """
    def __init__(self, labels: List):
        self.labels = labels
        # {doc_name: {sent_id: [mentions]}}
        self.labels_by_doc = {}
        self.group_labels()

    def group_labels(self):
        """
        A function to group all labels by docs and sentences.
        """
        for label in self.labels:
            label = Mention(**label)
            s_id = label.sentence_id
            d_name = label.doc_id
            if d_name not in self.labels_by_doc:
                self.labels_by_doc[d_name] = {s_id: [label]}
            elif s_id not in self.labels_by_doc[d_name]:
                self.labels_by_doc[d_name][s_id] = [label]
            else:
                self.labels_by_doc[d_name][s_id].append(label)
