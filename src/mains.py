import os
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
import configu

# Load embedding model
embedding_model = SentenceTransformer(configu.EMBEDDING_MODEL)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=configu.CHROMA_CLIENT_PATH)
collection = chroma_client.get_or_create_collection(name=configu.COLLECTION_NAME)

def extract_text_from_pdf(pdf_path):
    text = ""
    if not os.path.exists(pdf_path): 
        raise FileNotFoundError(f"❌ The file '{pdf_path}' does not exist.")
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for i, page in enumerate(reader.pages):
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    if not text.strip():
        raise ValueError(f"⚠️ No text could be extracted from '{pdf_path}'.")
    return text

def get_pdf_hash(pdf_path):
    sha256_hash = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(8192):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def store_pdf_in_vector_db(pdf_paths):
    for pdf_path in pdf_paths:
        #print(f"\n Processing PDF: {os.path.basename(pdf_path)}...")
        pdf_hash = get_pdf_hash(pdf_path)
        text = extract_text_from_pdf(pdf_path)
        embedding = embedding_model.encode([text])[0]
        collection.upsert(
            ids=[os.path.basename(pdf_path)],
            embeddings=[embedding.tolist()],
            documents=[text]
        )
        print(f"Stored content from '{os.path.basename(pdf_path)}' in ChromaDB!") 

def retrieve_entire_pdf_text_by_filename(filename):
    print(f"\n Retrieving document for: {filename}")
    result = collection.get(ids=[filename])
    if result and "documents" in result and result["documents"]:
        full_text = result["documents"][0]
        print(" Successfully retrieved text.\n")
        return full_text
    else:
        print(f"No content found in the vector DB for '{filename}'")
        return ""

if __name__ == "__main__":
    # Paths to both PDFs
    pdf1 = "C:/Users/DELL/Desktop/assis/src/html_ref.pdf"
    pdf2 = "C:/Users/DELL/Desktop/assis/src/html2.pdf"

    # Step 1: Store both PDFs in vector DB
    store_pdf_in_vector_db([pdf1, pdf2])

    # Step 2: Retrieve and print preview of both
    for pdf_file in [pdf1, pdf2]:
        filename = os.path.basename(pdf_file)
        text = retrieve_entire_pdf_text_by_filename(filename)
        if text:
            print(f"\nPreview of '{filename}':\n")
            print(text)
           
