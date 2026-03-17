from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import json

with open("data/train.json") as f:
    data = json.load(f)

texts = [x[0] for x in data]

dataset = Dataset.from_dict({"text": texts})

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize)

model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=8)

training_args = TrainingArguments(
    output_dir="model/bert_model",
    num_train_epochs=3,
    per_device_train_batch_size=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
