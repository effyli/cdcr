import os

import pickle
import argparse
import pyhocon

from transformers import AutoModel, AutoTokenizer
import torch

from cdcr.dataset import SeqDataset, fetch_dataloader
from cdcr.model import build_model, CDCRModel
from cdcr.utils.evaluation import Evaluator
from cdcr.dataset.vocab import build_vocab, Vocab
from cdcr.utils.ops import safe_div


def calculate_loss(model_out, targets):
    labels = targets['labels']
    seq_lens = targets['num_tokens']
    actions = targets['actions']

    loss = 0
    criterion = torch.nn.NLLLoss(reduction='sum')
    bi_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    log_probs = model_out["action_probs"]
    outputs = model_out["log_outputs"]
    # calculate per sentence
    for output, label, seq_len, log_prob, action in zip(outputs, labels, seq_lens, log_probs, actions):
        for idx in range(seq_len):
            if not bool(action[idx]):
                loss += criterion(output[idx, :].unsqueeze(0), label[idx].unsqueeze(0))
        # loss += criterion(output[:seq_len], label[:seq_len])
        # loss += bi_criterion(log_prob[:seq_len], action[:seq_len])

    # loss per token / sentence?
    # TODO: loss per mention?
    # loss /= sum(seq_lens).item()
    return loss


def train(dataset: SeqDataset,
          model: CDCRModel,
          vocab: Vocab,
          config: list,
          copy_id: int,
          device: torch.device,
          num_epochs: int,
          batch_size: int,
          learning_rate: float,
          val_dataset: SeqDataset = None):

    print("training...")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9)

    data_loader = fetch_dataloader(dataset=dataset, split="train", device=device, batch_size=batch_size)
    # for debugging dataset
    # for inputs, targets in data_loader:
    #     continue
    best_epoch_loss = float('+inf')
    best_model = None
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            # get outputs
            outputs = model(inputs, targets)
            # get loss
            loss = calculate_loss(outputs, targets)
            epoch_loss += loss

            # if epoch == 0:
            #     for i in range(batch_size):
            #         print("batch sample no.:", i)
            #         predicted_labels = outputs['outputs_ids'][i, 0, :].cpu()
            #         predicted_actions = outputs['action_probs']
            #         preds, ls = [], []
            #         for pred_l, l, a in zip(predicted_labels, targets['labels'][0], targets['actions'][0]):
            #             if a != torch.tensor([1]):
            #                 preds.append(pred_l)
            #                 ls.append(l)
            #             else:
            #                 preds.append(0)
            #                 ls.append(0)
            #         print("labels: ")
            #         print(torch.tensor(ls))
            #         print("predicted labels: ")
            #         print(torch.tensor(preds))
            #         print("actions: ")
            #         print(targets['actions'])

            # backprop
            loss.backward()
            # print(model.decision_making.weight.grad)
            # print(model.input_end_attn.weight.grad)
            optimizer.step()
            # _ = evaluate(dataset=val_dataset, model=model, device=device, batch_size=2, copy_id=copy_id)

        epoch_loss /= len(dataset)
        print("Epoch %d - Train Loss: %0.2f" % (epoch, epoch_loss))
        # validation
        val_epoch_loss = evaluate(dataset=val_dataset, model=model, copy_id=copy_id, device=device, batch_size=1, val_step=config.val_step)
        if val_epoch_loss < best_epoch_loss:
            best_epoch_loss = val_epoch_loss
            best_model = model.state_dict()
    if best_model:
        model.load_state_dict(best_model)


def evaluate(dataset: SeqDataset,
             model: CDCRModel,
             copy_id: int,
             device: torch.device,
             batch_size: int,
             val_step: int = 1000):
    """
    Evaluate on batched examples.
    Args:
        dataset: validation/test dataset used for evaluation

    """
    # initialize a evaluator for related metrics
    evaluator = Evaluator(total_steps=len(dataset), batch_size=batch_size, copy_id=copy_id, report_step=val_step)
    corrects = 0
    all_labels = 0
    model.eval()
    data_loader = fetch_dataloader(dataset=dataset, split="val", device=device, batch_size=batch_size)
    for inputs, targets in data_loader:
        model_out = model(inputs, targets)
        batch_loss = calculate_loss(model_out, targets)
        predicted_labels = model_out['outputs_ids'][0, 0, :].cpu()
        labels = targets['labels'].cpu().squeeze(0)
        actions = targets['actions'].cpu().squeeze(0)
        for pred, label, action in zip(predicted_labels, labels, actions):
            # if not copy
            if not torch.eq(action, torch.tensor(1.)):
                all_labels += 1
                corrects += int(torch.eq(pred, label))
        # calculate P,R,F1 score per batch
        # evaluator.update(predicted_labels, labels, batch_loss, targets['num_tokens'].cpu())
        # if evaluator.is_report():
        #     step_loss, step_acc, step_recall, step_precision, step_f_1 = evaluator.report()
        #     print("Eval step {}, out of {}".format(evaluator.step, len(dataset)))
        #     print("Accuracy of current {} samples is {}, recall is {}, precision is {}, f1 is {}".format(val_step, step_acc, step_recall, step_precision, step_f_1))
        #     print("Loss of current {} samples is {}".format(val_step, step_loss / val_step))
        torch.cuda.empty_cache()
    acc = safe_div(corrects, all_labels)
    print("overall accuracy is ", acc)
    model.train()
    return acc


if __name__ == '__main__':
    seed = 816
    torch.manual_seed(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # arguments parsing
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file(args.config)
    # set up GPU
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # model params
    batch_size = config.batch_size
    encoder = config.encoder
    decoder = config.decoder
    hidden_size = config.hidden_size
    # TODO: change size in pre-trained bert
    bert_size = config.bert_size
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    # loading spanBert tokenizer
    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")

    if 'ecb' in config.train_data:
        vocab_name = config.vocab_type
        if not os.path.exists(config.vocab_path + vocab_name):
            build_vocab(config)
        # loading pre-built vocab
        with open(config.vocab_path + vocab_name, 'rb') as f:
            vocab = pickle.load(f)
        train_labels = config.train_data_mentions
    elif 'ontoNotes' in config.train_data:
        vocab = None
        train_labels = None

    if vocab:
        sos_id = vocab["<sos>"]
        eos_id = vocab["<eos>"]
        copy_id = vocab["<copy>"]
    else:
        # pre-define a list of special ids
        # in Bert tokenizer vocab, vocab['[PAD]'] = 0, and from key 1 to 99 are unused
        # here, the sos_id is relative id in the sentence
        sos_id, eos_id, copy_id = 0, 2, 3

    # building dataset
    train_data = SeqDataset(data_path=config.train_data,
                            tokenizer=tokenizer,
                            label_path=train_labels,
                            vocab=vocab)
    if vocab:
        vocab_size = train_data.vocab.size
    else:
        vocab_size = None

    if "ecb" in config.val_data:
        val_labels = config.val_data_mentions
    elif "ontoNotes" in config.val_data:
        val_labels = None

    val_data = SeqDataset(data_path=config.val_data,
                          tokenizer=tokenizer,
                          label_path=val_labels,
                          vocab=vocab)

    # building model
    # bert
    bert_model = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
    bert_model.eval()
    model = build_model(encoder_name=encoder,
                        decoder_name=decoder,
                        hidden_size=hidden_size,
                        input_size=bert_size,
                        vocab_size=vocab_size,
                        sos_id=sos_id,
                        eos_id=eos_id,
                        copy_id=copy_id,
                        pre_trained_emb=bert_model)
    model.to(device)
    train(dataset=train_data,
          model=model,
          device=device,
          vocab=vocab,
          config=config,
          copy_id=copy_id,
          num_epochs=num_epochs,
          batch_size=batch_size,
          learning_rate=learning_rate,
          val_dataset=val_data)

    if config.save_model:
        model.save(config.save_model + 'model')







