import torch
import numpy as np


class Evaluator:
    def __init__(self):
        self.predicts = []
        self.labels = []

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def get_acc(self):
        return (self.predicts == self.labels).float().sum()

    def get_tp(self):
        return




