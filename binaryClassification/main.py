from binaryClassification.trainer import Trainer
from binaryClassification.dataloader import Dataset
from transformers import GPTNeoForSequenceClassification
from binaryClassification.config import Config
from faultInjector.hookSetter import HookSetter as Hook
from compressor.compressor import Compressor

config = Config()

if __name__ == "__main__":
    model = GPTNeoForSequenceClassification.from_pretrained(config.path, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id
    # Compressor(model)
    model.to(config.device)
    # Hook(model)
    # print("model: {}".format(model))
    dataset = Dataset()
    trainer = Trainer(model)
    # trainer.train(dataset.train_loader)
    trainer.evaluate(dataset.test_loader)