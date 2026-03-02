def merge_entities(predictions):
    entities = []
    current_entity = ""
    current_label = ""

    for token, label in predictions:
        token = token.replace("##", "")

        if label.startswith("B-"):
            if current_entity:
                entities.append((current_entity, current_label))
            current_entity = token
            current_label = label[2:]

        elif label.startswith("I-") and current_entity:
            current_entity += " " + token

        else:
            if current_entity:
                entities.append((current_entity, current_label))
                current_entity = ""
                current_label = ""

    if current_entity:
        entities.append((current_entity, current_label))

    return entities