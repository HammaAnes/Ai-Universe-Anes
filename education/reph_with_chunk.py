import os

import re

def chunk_text(text, max_words=200):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    chunk = []

    for sentence in sentences:
        if sum(len(w.split()) for w in chunk) + len(sentence.split()) > max_words:
            chunks.append(" ".join(chunk))
            chunk = [sentence]
        else:
            chunk.append(sentence)

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def rephrase_text(text):
    input_text = f"paraphrase: {text} </s>"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def rephrase_chapter_file(input_file, output_file):
    with open(f"document\{input_file}", 'r', encoding='utf-8') as f:
        chapter_text = f.read()

    chunks = chunk_text(chapter_text, max_words=200)
    rephrased_chunks = []

    for i, chunk in enumerate(chunks):
        try:
            print(f"Rephrasing chunk {i+1}/{len(chunks)}")
            rephrased = rephrase_text(chunk)
            rephrased_chunks.append(rephrased)
        except Exception as e:
            print(f"Error in chunk {i+1}: {e}")
            rephrased_chunks.append(chunk)  # fallback to original

    # Save the full rephrased chapter
    with open(f"document\{output_file}", 'w', encoding='utf-8') as f_out:
        f_out.write("\n\n".join(rephrased_chunks))

    print(f"Saved rephrased chapter to {output_file}")


for i in range(1, 10):  # Assuming chapters are named chapter_1.txt ... chapter_9.txt
    input_file = f"chapter_{i}.txt"
    output_file = f"chapter_{i}_rephrased.txt"
    rephrase_chapter_file(input_file, output_file)
