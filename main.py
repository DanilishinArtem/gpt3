from dataloaderLocal import Dataset
from trainer import Trainer
from models import getLogicModel, getQuestionAnsweringModel


if __name__ == "__main__":
    model = getLogicModel()
    dataset = Dataset()
    trainer = Trainer(model)
    trainer.train(dataset.train_loader)
    # trainer.evaluate(dataset.test_loader)