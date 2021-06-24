from typing import List
import torch
import numpy as np

from .ops import sum_to_int, safe_div


class Evaluator:
    """
    a evaluator following B-cubed style for evaluation
    """
    def __init__(self, total_steps: int, batch_size: int, copy_id: int, report_step: int = 100):
        self.batch_size = batch_size
        self.total_loss = 0
        self.step = 0
        self.report_step = report_step
        self.total_steps = total_steps
        self.copy_id = copy_id

        # use for storing results that are useful for calculating P,R,F
        # each chain in a format of {"chain_id": [|H intersection R|, |H|, |R|]},
        # where |H| is the number of mentions in  hypothesis chain,
        # and |R| is the number of mentions in reference chain
        self.chains = {}
        # a boolean indicator shows when a mention i
        self.boolean_indicators = {"HR": [True, True, True], "H": [False, True, False], "R": [False, False, True]}
        # this is a dictionary to store chain_id with corresponding entity_id
        self.dictionary = {}

        # use for accuracy as a sequence labeling task
        self.num_correct_preds = 0
        self.num_mentions = 0

    def update(self, predicted_labels, labels, loss, num_tokens):
        """
        For dev set.
        A function to update stats, including:
        :param predicted_labels:
        :param labels:
        :param loss:
        :param num_tokens:
        :return:
        """
        self.step += self.batch_size
        self.total_loss += loss.item()
        # dealing with multiple batches
        for predicted_label, label, num_token in zip(predicted_labels, labels, num_tokens):
            for i in range(num_token):
                if label[i] != self.copy_id:
                    self.num_mentions += 1
                    # true positive
                    if predicted_label[i] == label[i]:
                        self.num_correct_preds += 1
                        self.__update_dict(int(predicted_label[i]), self.boolean_indicators['HR'])
                    # false
                    else:
                        self.__update_dict(int(predicted_label[i]), self.boolean_indicators['H'])
                        self.__update_dict(int(label[i]), self.boolean_indicators['R'])

    def __update_dict(self, token_id, bools: list):
        """
        A private method for updating the dictionary.

        :param token_id:
        :param bools: a list of boolean indicators for [H&R, H, R]
        :return:
        """
        if token_id not in self.dictionary:
            self.dictionary[token_id] = len(self.chains)
            chain_id = len(self.chains)
            self.chains[chain_id] = [int(x) for x in bools]
        else:
            chain_id = self.dictionary[token_id]
            self.chains[chain_id] = tuple(x + int(y) for x, y in zip(self.chains[chain_id], bools))

    def is_report(self):
        return (self.step % self.report_step == 0) or (self.step >= self.total_steps)

    def report(self):
        loss = self.total_loss
        acc = safe_div(self.num_correct_preds, self.num_mentions)

        p, r = self.get_pr()
        f_1 = 2 * safe_div(p*r, p+r)
        return loss, acc, p, r, f_1

    def get_pr(self):
        """
        Precision and Recall are weighted (even now) sum of precision and recall per chain
        Precision_per_chain = HR / H
        Recall_per_chain = HR / R
        """
        p_sum, r_sum = 0, 0
        for _, stats in self.chains.items():
            p_sum += safe_div(stats[0], stats[1])
            r_sum += safe_div(stats[0], stats[2])
        return safe_div(p_sum, len(self.dictionary)), safe_div(r_sum, len(self.dictionary))

# depecrated
class OldEvaluator:
    def __init__(self, total_steps: int, batch_size: int, ent_ids: List, report_step: int = 100):
        self.batch_size = batch_size
        self.total_loss = 0
        self.correct_preds = 0
        self.num_tokens = 0

        # used for accuracy
        self.total_steps = total_steps
        self.report_step = report_step
        self.step = 0
        self.step_correct_pred = 0
        self.step_loss = 0
        self.step_num_tokens = 0

        # useful for recall, precision
        self.golden_labels = 0
        self.step_golden_labels = 0
        self.all_preds = 0
        self.step_preds = 0
        self.step_correct_pred_labels = 0
        self.correct_preds_labels = 0

        self.ent_ids = ent_ids

    def update(self, predicted_labels, labels, loss, num_tokens):
        """
        A function to update total loss, step loss, step correct predicts, correct predicts, step.
        :param predicted_labels:
        :param labels:
        :param loss:
        :return:
        """
        self.step += self.batch_size
        self.step_num_tokens += num_tokens.sum().item()
        self.num_tokens += num_tokens.sum().item()
        batch_correct_preds_with_unk = 0
        batch_preds = 0
        batch_correct_preds_labels = 0
        batch_golden_labels = 0

        for predicted_label, label, num_token in zip(predicted_labels, labels, num_tokens):
            batch_correct_preds_with_unk += sum_to_int(predicted_label[: num_token] == label[: num_token])
            batch_golden_labels += sum_to_int(label[:num_token] != 0)
            batch_preds += sum_to_int(predicted_label[: num_token] != 0)
            for i in range(num_token):
                if label[i] in self.ent_ids and predicted_label[i] == label[i]:
                    batch_correct_preds_labels += 1

        self.step_correct_pred += batch_correct_preds_with_unk
        self.correct_preds += batch_correct_preds_with_unk

        self.step_preds += batch_preds
        self.all_preds += batch_preds

        self.step_golden_labels += batch_golden_labels
        self.golden_labels += batch_golden_labels

        self.step_correct_pred_labels += batch_correct_preds_labels
        self.correct_preds_labels += batch_correct_preds_labels

        self.step_loss += loss
        self.total_loss += loss

    def to_report(self):
        return (self.step % self.report_step == 0) or (self.step >= self.total_steps)

    def step_report(self):
        """
        When it reaches report steps, return step_loss, step_acc
        """
        step_loss = self.step_loss
        step_acc = safe_div(self.step_correct_pred, self.step_num_tokens)

        # get recall, precision

        step_recall = safe_div(self.step_correct_pred_labels, self.step_golden_labels)
        step_precision = safe_div(self.step_correct_pred_labels, self.step_preds)

        self.step_loss, self.step_correct_pred, self.step_num_tokens = 0, 0, 0
        self.step_preds, self.step_correct_pred, self.step_golden_labels, self.step_correct_pred_labels = 0, 0, 0, 0
        return step_loss, step_acc, step_recall, step_precision

    def final_report(self):
        """
        Return final stats in the order of loss, accuracy, recall, precision
        """
        return (safe_div(self.total_loss, self.total_steps)), \
               (safe_div(self.correct_preds, self.num_tokens)), \
               (safe_div(self.correct_preds_labels, self.golden_labels)), \
               (safe_div(self.correct_preds_labels, self.all_preds))




