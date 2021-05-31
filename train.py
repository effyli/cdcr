import argparse
import pyhocon

from transformers import AutoModel, AutoTokenizer
import torch

from cdcr.dataset import SeqDataset, fetch_dataloader
from cdcr.model import build_model, CDCRModel
from cdcr.utils.evaluation import calculate_prf


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

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9)

    data_loader = fetch_dataloader(dataset=dataset, split="train", device=device, batch_size=batch_size)
    train_losses, val_losses = [], []
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
            evaluate(dataset=val_dataset, model=model, device=device, batch_size=1)

        epoch_loss /= len(dataset)
        print("Epoch %d - Train Loss: %0.2f" % (epoch, epoch_loss))
        # validation
        evaluate(dataset=val_dataset, model=model, device=device, batch_size=1)


def evaluate(dataset: SeqDataset,
             model: CDCRModel,
             device: torch.device,
             batch_size: int,
             val_step: int = 1000):
    """
    Evaluate on batched examples.
    Args:
        dataset: validation/test dataset used for evaluation

    """

    model.eval()
    total_loss = 0
    data_loader = fetch_dataloader(dataset=dataset, split="val", device=device, batch_size=batch_size)
    accs = []
    step_correct_pred = 0
    step = 0
    step_loss = 0
    for inputs, targets in data_loader:
        step += batch_size
        outputs = model(inputs, targets)
        batch_loss = calculate_loss(outputs, targets)
        total_loss += batch_loss
        step_loss += batch_loss
        predicted_labels = torch.argmax(outputs, dim=2)
        # calculate P,R,F1 score per batch
        predicted_out = predicted_labels.detach().numpy().tolist()
        labels = targets['labels'].detach().numpy().tolist()
        step_correct_pred += int((predicted_labels == targets['labels']).float().sum().item())
        if step % val_step == 0:
            step_acc = step_correct_pred / val_step
            accs.append(step_acc)
            print("Current Accuracy of {} samples is {}".format(val_step, step_acc))
            step_correct_pred = 0
            print("Current loss of {} samples is {}".format(val_step, step_loss / val_step))
            step_loss = 0

    print("Overall accuracy for all val data: {}, loss is: {}".format(sum(accs)/len(accs), total_loss/len(dataset)))
    model.train()


if __name__ == '__main__':
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

    # building dataset
    train_data = SeqDataset(data_path=config.train_data,
                            tokenizer=tokenizer,
                            label_path=config.train_data_mentions)
    vocab_size = train_data.entVocab.size

    val_data = SeqDataset(data_path=config.val_data,
                          tokenizer=tokenizer,
                          label_path=config.val_data_mentions,
                          entities_vocab=train_data.entVocab)

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







