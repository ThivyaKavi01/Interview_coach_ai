from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import random

app = FastAPI()

# Enable CORS (for connecting frontend if needed later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load questions from JSON
with open("qa_unit1.json", "r") as f:
    quiz_data = json.load(f)

# Track asked questions
asked_questions = []

@app.get("/")
def home():
    return {"message": "Quiz API is running!"}

@app.get("/question/")
def ask_next_question():
    remaining = list(set(quiz_data.keys()) - set(asked_questions))
    
    if not remaining:
        return {"message": "All questions have been asked!"}
    
    question = random.choice(remaining)
    asked_questions.append(question)
    
    return {"question": question}
