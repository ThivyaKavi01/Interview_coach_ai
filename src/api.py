from fast api import FastaPI , FIle, UploadFile, form 
from qa_model import generate_qa_pairs, store_pdf_in_vector_db, retrieve_entire_pdf_text_by_filename


app = FastAPI()

@app.post("/uploadfile/")
async def retrieve_entire_pdf_text_by_filename():
    