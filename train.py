import os

import pickle
import argparse
import pyhocon

from transformers import AutoModel, AutoTokenizer
import torch

from cdcr.dataset import SeqDataset, fetch_dataloader
from cdcr.model import build_model, CDCRModel
from cdcr.utils.evaluation import Evaluator
from cdcr.utils.vocab import EntVocab, build_vocab


def calculate_loss(outputs, targets):
    labels = targets['labels']
    seq_lens = targets['num_tokens']
    loss = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    # calculate per sentence
    for output, label, seq_len in zip(outputs, labels, seq_lens):
        loss += criterion(output[:seq_len], label[:seq_len])

    # loss per token / sentence?
    loss /= sum(seq_lens).item()
    return loss


def train(dataset: SeqDataset,
          model: CDCRModel,
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
            # get relation scores
            outputs = model(inputs, targets)
            # get loss
            loss = calculate_loss(outputs, targets)
            epoch_loss += loss
            # backprop
            loss.backward()
            optimizer.step()
            _ = evaluate(dataset=val_dataset, model=model, device=device, batch_size=2)

        epoch_loss /= len(dataset)
        print("Epoch %d - Train Loss: %0.2f" % (epoch, epoch_loss))
        # validation
        val_epoch_loss = evaluate(dataset=val_dataset, model=model, device=device, batch_size=1)
        if val_epoch_loss < best_epoch_loss:
            best_epoch_loss = val_epoch_loss
            best_model = model.state_dict()
    if best_model:
        model.load_state_dict(best_model)


def evaluate(dataset: SeqDataset,
             model: CDCRModel,
             device: torch.device,
             batch_size: int,
             val_step: int = 100):
    """
    Evaluate on batched examples.
    Args:
        dataset: validation/test dataset used for evaluation

    """
    # initialize a evaluator for related metrics
    evaluator = Evaluator(total_steps=len(dataset), batch_size=batch_size, report_step=val_step)

    model.eval()
    data_loader = fetch_dataloader(dataset=dataset, split="val", device=device, batch_size=batch_size)
    for inputs, targets in data_loader:
        outputs = model(inputs, targets)
        batch_loss = calculate_loss(outputs, targets)
        predicted_labels = torch.argmax(outputs.cpu(), dim=2)
        labels = targets['labels'].cpu()
        # calculate P,R,F1 score per batch
        evaluator.update(predicted_labels, labels, batch_loss.item(), targets['num_tokens'].cpu())
        if evaluator.to_report():
            step_loss, step_acc, step_recall, step_precision = evaluator.step_report()
            print("Eval step {}, out of {}".format(evaluator.step, len(dataset)))
            print("Accuracy of current {} samples is {}, recall is {}, precision is {}".format(val_step, step_acc, step_recall, step_precision))
            print("Loss of current {} samples is {}".format(val_step, step_loss / val_step))
        torch.cuda.empty_cache()
    total_loss, total_acc, total_recall, total_precision = evaluator.final_report()
    print("Overall accuracy for all val data: {}, loss is: {}, recall is {}, precision is {}".format(total_acc, total_loss, total_recall, total_precision))
    model.train()
    return total_loss


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

    # build_vocab(config)
    # loading pre-built vocab
    with open(config.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # building dataset
    train_data = SeqDataset(data_path=config.train_data,
                            tokenizer=tokenizer,
                            label_path=config.train_data_mentions,
                            entities_vocab=vocab)
    vocab_size = train_data.entVocab.size

    val_data = SeqDataset(data_path=config.val_data,
                          tokenizer=tokenizer,
                          label_path=config.val_data_mentions,
                          entities_vocab=vocab)


    # building model
    # bert
    bert_model = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
    bert_model.eval()
    model = build_model(encoder_name=encoder,
                        decoder_name=decoder,
                        hidden_size=hidden_size,
                        input_size=bert_size,
                        vocab_size=vocab_size,
                        pre_trained_emb=bert_model)
    model.to(device)
    train(dataset=train_data,
          model=model,
          device=device,
          num_epochs=num_epochs,
          batch_size=batch_size,
          learning_rate=learning_rate,
          val_dataset=val_data)

    if config.save_model:
        model.save(config.save_model)







