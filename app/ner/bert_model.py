from transformers import pipeline

try:
    ner_pipeline = pipeline("ner", model="model/bert_model", tokenizer="model/bert_model")
except:
    ner_pipeline = pipeline("ner")

def extract_entities_bert(text):
    results = ner_pipeline(text)

    entities = {"PARTY": [], "DATE": [], "AMOUNT": [], "CLAUSE": []}

    for r in results:
        label = r.get("entity", "")
        word = r.get("word", "")
        if label in entities:
            entities[label].append(word)

    return entities