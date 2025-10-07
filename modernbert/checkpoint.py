import torch
import torch_neuronx
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "bert-base-cased-finetuned-mrpc"

model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
torch.save({"model":model.state_dict()}, "bert.pt")