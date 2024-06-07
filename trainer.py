import torch
import torch.optim as optim
from config import Config

config = Config()


class Trainer:
    def __init__(self, model: torch.nn.Module):
        self.model = model.to(config.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        for i in range(config.epochs):
            step = 0
            for batch in train_loader:
                step += 1
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].float().to(config.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                logits_avg = outputs.mean(dim=1)

                loss = self.loss_fn(logits_avg, labels)

                # logits_avg_positive = logits_avg[:,1]
                # prob = torch.sigmoid(logits_avg_positive)
                # logits = (prob > 0.5)

                # loss = self.loss_fn(logits, labels)




                # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                # loss = outputs.loss

                # logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                # logits = logits.argmax().item()
                # # logits = outputs.logits
                # loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                print("Loss for epoch " + str(i) + " step " + str(step) + ": " + str(loss.item() / config.batch_size))
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        return avg_loss, accuracy