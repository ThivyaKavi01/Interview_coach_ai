import os
import json
import re
from transformers import pipeline
import configu
from mains import store_pdf_in_vector_db, retrieve_entire_pdf_text_by_filename

# Load the text generation model
qa_pipeline = pipeline("text-generation", model=configu.TEXT_GEN_MODEL)

def generate_qa_pairs(text, unit, num_pairs=12):
    """Generate Q&A pairs using TinyLlama."""
    if not text:
        print("No text to generate from.")
        return {}, ""

    max_input_len = 3000
    if len(text) > max_input_len:
        text = text[:max_input_len]

    print(f"\n🤖 Generating {num_pairs} Q&A pairs for Unit {unit}...\n")

    prompt = f"""
You are reading notes from Unit {unit}. Read the following text and generate {num_pairs} thoughtful and varied question-answer pairs. 
Format it clearly like this:

Q1: What is HTML?
A1: HTML stands for HyperText Markup Language...

Q2: ...
A2: ...

Text:
{text}

Q&A:
"""

    result = qa_pipeline(prompt, max_new_tokens=1500)[0]["generated_text"]
    return parse_qa_output(result), result


def parse_qa_output(raw_text):
    """Improved parser using regex for robust Q&A extraction"""
    qa_dict = {}
    matches = re.findall(r"(Q\d+:\s*)(.+?)(?:\nA\d+:\s*)(.+?)(?=\nQ\d+:|\Z)", raw_text, re.DOTALL)

    for q_prefix, question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        if question and answer:
            qa_dict[question] = answer

    if not qa_dict:
        print(" No Q&A pairs were parsed. Check the format or raw text.")
    return qa_dict


def save_to_txt(raw_text, unit):
    filename = f"qa_unit{unit}_raw.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f" Raw Q&A saved to {filename}")


def save_to_json(qa_dict, unit):
    filename = f"qa_unit{unit}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(qa_dict, f, indent=4, ensure_ascii=False)
    print(f" Structured Q&A saved to {filename}")


def preview_qa(qa_dict, count=3):
    print("\n Sample Q&A Preview:\n")
    for i, (q, a) in enumerate(qa_dict.items()):
        print(f"Q{i+1}: {q}\nA{i+1}: {a}\n")
        if i + 1 == count:
            break


if __name__ == "__main__":
    print(" Starting full Q&A generation for all units...\n")

    pdfs = {
        "1": "html_ref.pdf",
        "2": "html2.pdf"
    }

    for unit, filename in pdfs.items():
        print(f"\n Processing PDF: {filename}...\n")
        pdf_path = f"C:/Users/DELL/Desktop/assis/src/{filename}"

        if not os.path.exists(pdf_path):
            print(f" File not found: {pdf_path}")
            continue

        store_pdf_in_vector_db([pdf_path])

        retrieved_text = retrieve_entire_pdf_text_by_filename(filename)

        if not retrieved_text:
            print(f" No content retrieved for {filename}. Skipping...")
            continue

        print(f"\n First 300 characters of Unit {unit} text:\n{retrieved_text[:300]}\n")

        qa_dict, raw_text = generate_qa_pairs(retrieved_text, unit, num_pairs=12)

        if qa_dict:
            save_to_txt(raw_text, unit)
            save_to_json(qa_dict, unit)
            preview_qa(qa_dict)

    print("\n All done! Q&A generated and saved for all units.")
