import argparse
import pyhocon

from transformers import AutoModel, AutoTokenizer
import torch

from cdcr.dataset import SeqDataset
from cdcr.encoder import IndependentEncoder

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

    # loading spanBert
    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")

    # building model
    # bert
    bert_model = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
    bert_model.eval()

    #testing encoder
    encoder = IndependentEncoder(pre_trained_emb=bert_model)

    # building dataset
    train_data = SeqDataset(data_path=config.train_data, tokenizer=tokenizer, label_path=config.train_data_mentions)
    data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=lambda samples: train_data.batch_fn(samples, device),
                                              num_workers=0)

    for inputs, targets in data_loader:
        outputs = encoder(inputs)




