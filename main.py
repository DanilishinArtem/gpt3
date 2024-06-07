from dataloaderLocal import Dataset
from trainer import Trainer
from models import getLogicModel, getQuestionAnsweringModel
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from config import Config


config = Config()

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def run(model, train_dataset, eval_dataset, tokenizer, ):
    training_args = TrainingArguments(
        output_dir="gpt3",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=0.01,
        load_best_model_at_end=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()



if __name__ == "__main__":
    model = getLogicModel()
    dataset = Dataset()
    
    data_collator = DataCollatorWithPadding(tokenizer=dataset.tokenizer)
    run(model, dataset.train_loader, dataset.test_loader, dataset.tokenizer)


    # trainer = Trainer(model)
    # trainer.train(dataset.train_loader)
    # trainer.evaluate(dataset.test_loader)