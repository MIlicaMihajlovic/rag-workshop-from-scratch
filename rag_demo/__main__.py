### Utility libraries
import argparse
import os
import time
from dotenv import load_dotenv
import requests
import numpy as np

### PostgreSQL adapter for Python
import psycopg

### PyPDF for text extraction
from PyPDF2 import PdfReader

### Transformers for local embeddings and QA
from transformers import AutoTokenizer, AutoModel, pipeline
import torch

### Constants
load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CHUNK_SIZE = 2048
EMBEDDINGS_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
MODEL_API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
hf_api_key = os.environ.get("HF_API_KEY")
HEADERS = {
    "Authorization": f"""Bearer {hf_api_key}""",
    "Content-Type": "application/json",
    "x-wait-for-model": "true",
}

# Load local embedding model
print("Loading embedding model...")
embedding_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
embedding_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
embedding_model.eval()

# Load local QA pipeline
print("Loading QA model...")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if torch.cuda.is_available() else -1)

### Argument parser
parser = argparse.ArgumentParser(description="RAG Demo")
parser.add_argument(
    "--skip-embedding-step",
    action="store_true",
    help="Skip the embedding step and use the existing embeddings if this flag is provided.",
)
parser.add_argument(
    "--use-remote-api",
    action="store_true",
    help="Use remote Hugging Face API instead of local models (requires internet).",
)
args = parser.parse_args()

### Useful functions [can go to a utils.py file]
def get_embedding_local(text):
    """Get embedding using local model"""
    inputs = embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy().tolist()

def get_embedding(payload):
    """Get embedding - uses local model by default, falls back to API if specified"""
    if args.use_remote_api:
        response = requests.post(
            EMBEDDINGS_API_URL,
            headers=HEADERS,
            json=payload,
        )
        return response.json()
    else:
        return get_embedding_local(payload)

def get_answer_local(context, question):
    """Get answer using local QA model"""
    result = qa_pipeline(question=question, context=context)
    return {"answer": result["answer"]}

def get_answer(payload):
    """Get answer - uses local model by default, falls back to API if specified"""
    if args.use_remote_api:
        response = requests.post(
            MODEL_API_URL,
            headers=HEADERS,
            json=payload,
        )
        return response.json()
    else:
        # For local model, payload structure is different
        return get_answer_local(payload["context"], payload["question"])
    return response.json()


### PostgreSQL database url and connection
database_url = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:6432/rag_demo"
)
db = psycopg.Connection.connect(database_url)


# This is very naive chunking, just to show the concept. LangChain/LlamaIndex have excellent chunking libraries.  Do not use
# this technique in production as it will yield very bad results.
def split_string_by_length(input_string, length):
    return [input_string[i : i + length] for i in range(0, len(input_string), length)]


# Loop through chunks from the pdf and create embeddings in the database

if not args.skip_embedding_step:
    print("Cleaning database...")
    db.execute("TRUNCATE TABLE chunks")

    tic = time.perf_counter()
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)

        reader = PdfReader(file_path)
        content = ""
        for page in reader.pages:
            content += page.extract_text()

        for chunk in split_string_by_length(content, CHUNK_SIZE):
            print(f"Creating embedding for chunk: {chunk[0:20]}...")
            
            db.execute(
                "INSERT INTO chunks (embedding, chunk) VALUES (%s, %s)",
                [str(get_embedding(chunk)), chunk],
            )

        print(f"\nTotal index time: {time.perf_counter() - tic}ms")
        db.commit()

question = input("\nEnter question: ")

# Create embedding from question.  Many RAG applications use a query rewriter before querying
# the vector database.  For more information on query rewriting, see this whitepaper:
#    https://arxiv.org/abs/2305.14283
question_embedding = get_embedding(question)

result = db.execute(
    "SELECT (embedding <=> %s::vector)*100 as score, chunk FROM chunks ORDER BY score DESC LIMIT 5", 
    (question_embedding,)
)

rows = list(result)

print("scores: ", [row[0] for row in rows])
context = "\n\n".join([row[1] for row in rows])

prompt = f"""
Answer the question using only the following context:

{context}

Question: {question}
"""

if args.use_remote_api:
    answer = get_answer(
        {
            "inputs": {
                "question": question,
                "context": context,
            }
        })
else:
    answer = get_answer(
        {
            "question": question,
            "context": context,
        }
    )

print(f"\nUsing {len(rows)} chunks in answer. Answer:\n")
if isinstance(answer, dict) and "answer" in answer:
    print(answer["answer"])
else:
    print(answer)

view_prompt = input("\nWould you like to see the raw prompt? [Y/N] ")
if view_prompt == "Y":
    print("\n" + prompt)