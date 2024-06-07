import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import Config

config = Config()


# Разделим на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split

  
class IMDbDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['review']
        label = 1 if self.dataframe.iloc[idx]['label'] == 'pos' else 0
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class Dataset:
    def __init__(self) -> None:
        train_file = config.path
        df = pd.read_csv(train_file, encoding='ISO-8859-1')
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        tokenizer = AutoTokenizer.from_pretrained("gpt-neo-1.3B")
        tokenizer.pad_token = tokenizer.eos_token
        train_dataset = IMDbDataset(train_df, tokenizer, max_len=256)
        test_dataset = IMDbDataset(test_df, tokenizer, max_len=256)

        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=config.batch_size)