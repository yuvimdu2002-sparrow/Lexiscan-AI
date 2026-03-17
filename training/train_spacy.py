import spacy
import json
from spacy.training import Example
import os # Import the os module

with open("train.json") as f:
    TRAIN_DATA = json.load(f)

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# Corrected iteration: each 'training_example' is now a dictionary
for training_example in TRAIN_DATA:
    # Access the 'entities' list from the dictionary
    for ent in training_example["entities"]:
        ner.add_label(ent['label']) # Corrected from ent[2] to ent['label']

optimizer = nlp.begin_training()

for epoch in range(20):
    losses = {}
    # Corrected iteration: each 'training_example' is now a dictionary
    for training_example in TRAIN_DATA:
        text = training_example["text"]
        # Convert list of entity dictionaries to list of (start, end, label) tuples
        entities_tuples = [
            (ent['start'], ent['end'], ent['label'])
            for ent in training_example["entities"]
        ]
        # Create the annotations dictionary required by Example.from_dict
        annotations = {"entities": entities_tuples}

        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], losses=losses)
    print(f"Epoch {epoch} Loss: {losses}")

# Define the output directory path
output_dir = "model/spacy_model"

# Create the full target directory if it doesn't exist
# os.makedirs will create intermediate directories if they don't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

nlp.to_disk(output_dir)