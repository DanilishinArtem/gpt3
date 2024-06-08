import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from binaryClassification.config import Config

config = Config()

tokenizer = AutoTokenizer.from_pretrained(config.path)
dataset = load_dataset("imdb", split=['train', 'test'])

# Установка eos_token в качестве pad_token
tokenizer.pad_token = tokenizer.eos_token


class IMDbDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.dataset = tokenized_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'input_ids': self.dataset[idx]['input_ids'],
            'attention_mask': self.dataset[idx]['attention_mask'],
            'labels': self.dataset[idx]['labels']
        }
    

class Dataset:
    def __init__(self):
        dataset = load_dataset("imdb")
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        train_dataset = tokenized_datasets['train']
        test_dataset = tokenized_datasets['test']
        train_dataset = IMDbDataset(train_dataset)
        test_dataset = IMDbDataset(test_dataset)
        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    def tokenize_function(self, examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)
