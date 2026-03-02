from app.config import MAX_CHUNK_WORDS

def chunk_text(text):
    words = text.split()
    chunks = []

    for i in range(0, len(words), MAX_CHUNK_WORDS):
        chunk = " ".join(words[i:i+MAX_CHUNK_WORDS])
        chunks.append(chunk)

    return chunks