

import os
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import PyPDF2
import configu  # Your config file

# Load embedding model
embedding_model = SentenceTransformer(configu.EMBEDDING_MODEL)

# Initialize ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(path=configu.CHROMA_CLIENT_PATH)
collection = chroma_client.get_or_create_collection(name=configu.COLLECTION_NAME)

# Load the TinyLlama model for text generation
qa_pipeline = pipeline("text-generation", model=configu.TEXT_GEN_MODEL)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    if not os.path.exists(pdf_path): 
        raise FileNotFoundError(f"Error: The file '{pdf_path}' does not exist.")
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    if not text.strip():
        raise ValueError(f"Error: No text could be extracted from '{pdf_path}'.")
    return text

def get_pdf_hash(pdf_path):
    """Generate a unique hash for the PDF file based on its content."""
    sha256_hash = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(8192):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def store_pdf_in_vector_db(pdf_paths):
    """Stores text from PDFs as embeddings in ChromaDB using upsert to avoid duplication."""
    for pdf_path in pdf_paths:
        pdf_hash = get_pdf_hash(pdf_path)
        text = extract_text_from_pdf(pdf_path)
        embedding = embedding_model.encode([text])[0]
        collection.upsert(
            ids=[pdf_hash],  # Using PDF hash as a unique ID
            embeddings=[embedding.tolist()],
            documents=[text]
        )
        print(f"Stored content from '{os.path.basename(pdf_path)}' in ChromaDB!") 

def retrieve_entire_pdf_text():
    """Retrieve the entire text of the stored PDF from the vector database."""
    result = collection.get()
    if result and "documents" in result and result["documents"]:
        full_text = result["documents"][0]
        print("\nRetrieved Full Text from Vector Database.\n")
        return full_text
    else:
        print("No stored documents found in the vector database.")
        return ""

def generate_questions(text):
    """Generate 15 questions based on the given text using TinyLlama."""
    if not text:
        print("No text available for question generation.")
        return

    # Optional: truncate long input for performance
    max_input_len = 3000
    if len(text) > max_input_len:
        text = text[:max_input_len]

    print("\nGenerating 15 questions...\n")

    prompt = f"""
Read the following text and generate 15 thoughtful and varied questions based on the content. 
The questions can be factual, conceptual, or critical thinking based.

Text:
{text}

Questions:
"""
    questions = qa_pipeline(prompt, max_new_tokens=200)
    print("\nGenerated Questions:\n")
    print(questions[0]['generated_text'])

    # Optional: Save to file
    with open("generated_questions.txt", "w", encoding="utf-8") as f:
        f.write(questions[0]['generated_text'])

if __name__ == "__main__":
    # Replace this with your actual PDF path
    retrieve_pdf_path = "C:/Users/DELL/Desktop/thiv/html_ref.pdf"

    # Step 1: Store in ChromaDB
    store_pdf_in_vector_db([retrieve_pdf_path])

    # Step 2: Retrieve text
    retrieved_text = retrieve_entire_pdf_text()

    # Step 3: Generate 15 questions
    generate_questions(retrieved_text)
