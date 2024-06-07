import torch


class Config:
    def __init__(self):
        self.epochs = 10
        self.batch_size = 4
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = "/home/adanilishin/gpt3/imdb-dataset/imdb_master.csv"