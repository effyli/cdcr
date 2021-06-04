import torch
import numpy as np

from .ops import sum_to_int, safe_div


class Evaluator:
    def __init__(self, total_steps: int, batch_size: int, report_step: int = 100):
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
        self.num_tokens += num_tokens
        batch_correct_preds_with_unk = 0
        batch_preds = 0
        batch_correct_preds_labels = 0
        batch_golden_labels = 0

        for predicted_label, label, num_token in zip(predicted_labels, labels, num_tokens):
            batch_correct_preds_with_unk += sum_to_int(predicted_label[: num_token] == label[: num_token])
            batch_golden_labels += sum_to_int(label[:num_token] != 0)
            batch_preds += sum_to_int(predicted_label[: num_token] != 0)
            for i in range(num_token):
                if label[i] != 0 and predicted_label[i] == label[i]:
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
        return self.total_loss, \
               (safe_div(self.correct_preds, self.step_num_tokens)), \
               (safe_div(self.correct_preds_labels, self.golden_labels)), \
               (safe_div(self.correct_preds_labels, self.all_preds))




