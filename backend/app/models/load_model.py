from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from app.config import MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=7)
model.eval()