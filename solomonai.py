import ollama
import time
import os
import json
import numpy as np
from numpy.linalg import norm


# Parse a single file and return its paragraphs
def parse_file(filepath):
    with open(filepath, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
        if buffer:
            paragraphs.append(" ".join(buffer))
        return paragraphs

def chunk_text(paragraphs, max_words=200):
    """
    Splits paragraphs into smaller chunks of text with a maximum number of words.

    Args:
        paragraphs (list of str): List of paragraphs to split.
        max_words (int): Maximum number of words per chunk.

    Returns:
        list of str: List of text chunks.
    """
    chunks = []
    for paragraph in paragraphs:
        words = paragraph.split()
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
    return chunks
# Parse all text files in a folder and return paragraphs from all files
def parse_folder(folder_path):
    all_paragraphs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}")
            all_paragraphs.extend(parse_file(filepath))
    return all_paragraphs


# Save embeddings to a file
def save_embeddings(filename, embeddings):
    os.makedirs("embeddings", exist_ok=True)
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)


# Load embeddings from a file
def load_embeddings(filename):
    filepath = f"embeddings/{filename}.json"
    if not os.path.exists(filepath):
        return False
    with open(filepath, "r") as f:
        return json.load(f)


# Generate or load embeddings for a set of text chunks
def get_embeddings(filename, modelname, chunks):
    embeddings = load_embeddings(filename)
    if embeddings is not False:
        return embeddings

    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings


# Find the most similar chunks to a given embedding using cosine similarity
def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)


def main():

    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        Context:
    """

    folder_path = "./Text"  # Folder containing .txt files
    embeddings_filename = "all_context_embeddings"
    max_chunk_words = 200 
    
    # Parse all text files in the folder
    paragraphs = parse_folder(folder_path)

    chunks = chunk_text(paragraphs, max_words=max_chunk_words)
    print(f"Total chunks created: {len(chunks)}")


    # Generate or load embeddings
    embeddings = get_embeddings(embeddings_filename, "nomic-embed-text:latest",chunks)

    while True:
        prompt = input("What do you want to know? -> ")
        if not prompt.strip():
            print("Exiting...")
            break

        # Generate embedding for the prompt
        prompt_embedding = ollama.embeddings(model="nomic-embed-text:latest", prompt=prompt)["embedding"]

        # Find the most similar chunks
        most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:80]

        # Generate response using the most similar paragraphs
        context = "\n".join(chunks[item[1]] for item in most_similar_chunks)
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + context},
                {"role": "user", "content": prompt},
            ],
        )

        print("\n\n")
        print(response["message"]["content"])


if __name__ == "__main__":
    main()
