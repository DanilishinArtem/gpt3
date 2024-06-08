from transformers import GPTNeoForQuestionAnswering
from transformers import GPTNeoForCausalLM
from transformers import GPTNeoForSequenceClassification

def getLogicModel():
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    model = GPTNeoForCausalLM.from_pretrained("gpt-neo-1.3B", num_labels=2, id2label=id2label, label2id=label2id)
    return model

def getQuestionAnsweringModel():
    model = GPTNeoForQuestionAnswering.from_pretrained("gpt-neo-1.3B")
    return model

def gptSequenceClassificationModel():
    model = GPTNeoForSequenceClassification.from_pretrained("EleutherAI/gpt-neo-1.3B", num_labels=2)
    return model