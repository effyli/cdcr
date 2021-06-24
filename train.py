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


def calculate_loss(model_out, targets):
    labels = targets['labels']
    seq_lens = targets['num_tokens']
    actions = targets['actions']

    loss = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    bi_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    log_probs = model_out["log_probs"]
    outputs = model_out["outputs"]
    # calculate per sentence
    for output, label, seq_len, log_prob, action in zip(outputs, labels, seq_lens, log_probs, actions):
        loss += criterion(output[:seq_len], label[:seq_len])
        loss += bi_criterion(log_prob[:seq_len], action[:seq_len].float())

    # loss per token / sentence?
    # TODO: loss per mention?
    loss /= sum(seq_lens).item()
    return loss


def train(dataset: SeqDataset,
          model: CDCRModel,
          vocab: Vocab,
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
            # backprop
            loss.backward()
            optimizer.step()
            # _ = evaluate(dataset=val_dataset, model=model, device=device, batch_size=2, copy_id=copy_id)

        epoch_loss /= len(dataset)
        print("Epoch %d - Train Loss: %0.2f" % (epoch, epoch_loss))
        # validation
        val_epoch_loss = evaluate(dataset=val_dataset, model=model, copy_id=copy_id, device=device, batch_size=1)
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

    model.eval()
    data_loader = fetch_dataloader(dataset=dataset, split="val", device=device, batch_size=batch_size)
    for inputs, targets in data_loader:
        model_out = model(inputs, targets)
        outputs = model_out["outputs"]
        batch_loss = calculate_loss(model_out, targets)
        predicted_labels = torch.argmax(outputs.cpu(), dim=2)
        labels = targets['labels'].cpu()
        # calculate P,R,F1 score per batch
        evaluator.update(predicted_labels, labels, batch_loss, targets['num_tokens'].cpu())
        if evaluator.is_report():
            step_loss, step_acc, step_recall, step_precision, step_f_1 = evaluator.report()
            print("Eval step {}, out of {}".format(evaluator.step, len(dataset)))
            print("Accuracy of current {} samples is {}, recall is {}, precision is {}".format(val_step, step_acc, step_recall, step_precision))
            print("Loss of current {} samples is {}".format(val_step, step_loss / val_step))
        torch.cuda.empty_cache()
    model.train()
    return step_loss


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # arguments parsing
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file(args.config)
    # set up GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    vocab_name = config.vocab_type
    if not os.path.exists(config.vocab_path + vocab_name):
        build_vocab(config)
    # loading pre-built vocab
    with open(config.vocab_path + vocab_name, 'rb') as f:
        vocab = pickle.load(f)

    sos_id = vocab["<sos>"]
    eos_id = vocab["<eos>"]
    copy_id = vocab["<copy>"]

    # building dataset
    train_data = SeqDataset(data_path=config.train_data,
                            tokenizer=tokenizer,
                            label_path=config.train_data_mentions,
                            vocab=vocab)
    vocab_size = train_data.vocab.size

    val_data = SeqDataset(data_path=config.val_data,
                          tokenizer=tokenizer,
                          label_path=config.val_data_mentions,
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
          copy_id=copy_id,
          num_epochs=num_epochs,
          batch_size=batch_size,
          learning_rate=learning_rate,
          val_dataset=val_data)

    if config.save_model:
        model.save(config.save_model + 'model')







