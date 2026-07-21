### Utility libraries
import argparse
import os
import re
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
CHUNK_TOKEN_SIZE = 350
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
parser.add_argument(
    "--chunk-token-size",
    type=int,
    default=CHUNK_TOKEN_SIZE,
    help="Maximum tokens per chunk before overlap-based splitting is applied.",
)
parser.add_argument(
    "--chunk-overlap-ratio",
    type=float,
    default=0.15,
    help="Chunk overlap ratio used when splitting long passages.",
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


def clean_extracted_text(text):
    """Normalize PDF extraction artifacts while keeping paragraph boundaries."""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def looks_like_heading(line):
    """Heuristic heading detection for structure-aware chunking."""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 100:
        return False
    if stripped.endswith((".", "?", "!", ";")):
        return False

    alpha_chars = [char for char in stripped if char.isalpha()]
    if not alpha_chars:
        return False

    uppercase_ratio = sum(1 for char in alpha_chars if char.isupper()) / len(alpha_chars)
    title_case_words = [word for word in stripped.split() if word[:1].isupper()]
    title_case_ratio = len(title_case_words) / max(1, len(stripped.split()))

    starts_with_numbered_section = bool(re.match(r"^(\d+(\.\d+)*[\).]?)\s+", stripped))
    return starts_with_numbered_section or uppercase_ratio >= 0.6 or title_case_ratio >= 0.8


def token_count(text):
    """Count tokens using the embedding tokenizer for consistent chunk sizing."""
    return len(embedding_tokenizer(text, add_special_tokens=False)["input_ids"])


def split_long_text_with_token_overlap(text, max_tokens, overlap_tokens):
    """Token-level fallback splitter for oversized passages."""
    input_ids = embedding_tokenizer(text, add_special_tokens=False)["input_ids"]
    if not input_ids:
        return []

    stride = max(1, max_tokens - overlap_tokens)
    chunks = []
    start = 0

    while start < len(input_ids):
        end = min(start + max_tokens, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_text = embedding_tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(input_ids):
            break
        start += stride

    return chunks


def chunk_pdf_by_document_structure(file_path, max_chunk_tokens=CHUNK_TOKEN_SIZE, overlap_ratio=0.15):
    """Split a PDF into semantically coherent chunks using headings and paragraphs."""
    reader = PdfReader(file_path)
    pages_text = [clean_extracted_text(page.extract_text() or "") for page in reader.pages]
    full_text = "\n\n".join([page for page in pages_text if page])
    if not full_text:
        return []

    overlap_tokens = max(1, int(max_chunk_tokens * overlap_ratio))
    lines = [line.rstrip() for line in full_text.split("\n")]

    sections = []
    current_heading = "Document"
    current_lines = []

    for line in lines:
        if not line.strip():
            current_lines.append("")
            continue

        if looks_like_heading(line):
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append((current_heading, section_text))
                current_lines = []
            current_heading = line.strip()
            continue

        current_lines.append(line)

    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((current_heading, section_text))

    chunks = []
    for heading, section_text in sections:
        heading_prefix = f"{heading}\n\n" if heading else ""
        heading_tokens = token_count(heading_prefix)
        section_with_heading = f"{heading_prefix}{section_text}" if heading_prefix else section_text

        if token_count(section_with_heading) <= max_chunk_tokens:
            chunks.append(section_with_heading)
            continue

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
        paragraph_buffer = heading_prefix

        for paragraph in paragraphs:
            candidate = f"{paragraph_buffer}{paragraph}"
            if token_count(candidate) <= max_chunk_tokens:
                paragraph_buffer = f"{candidate}\n\n"
                continue

            if paragraph_buffer.strip():
                chunks.append(paragraph_buffer.strip())
                paragraph_buffer = heading_prefix

            candidate_with_heading = f"{heading_prefix}{paragraph}" if heading_prefix else paragraph
            if token_count(candidate_with_heading) <= max_chunk_tokens:
                paragraph_buffer = f"{paragraph_buffer}{paragraph}\n\n"
            else:
                text_budget = max(32, max_chunk_tokens - heading_tokens)
                long_parts = split_long_text_with_token_overlap(paragraph, text_budget, min(overlap_tokens, max(8, text_budget // 3)))
                if heading_prefix:
                    long_parts = [f"{heading_prefix}{part}" for part in long_parts]
                chunks.extend(long_parts)

        if paragraph_buffer.strip():
            chunks.append(paragraph_buffer.strip())

    return [chunk for chunk in chunks if chunk.strip()]


# Loop through chunks from the pdf and create embeddings in the database

if not args.skip_embedding_step:
    print("Cleaning database...")
    db.execute("TRUNCATE TABLE chunks")

    tic = time.perf_counter()
    for filename in os.listdir(DATA_DIR):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(DATA_DIR, filename)
        chunks = chunk_pdf_by_document_structure(
            file_path,
            max_chunk_tokens=args.chunk_token_size,
            overlap_ratio=args.chunk_overlap_ratio,
        )

        for chunk in chunks:
            print(f"Creating embedding for chunk: {chunk[0:20]}...")

            db.execute(
                "INSERT INTO chunks (embedding, chunk) VALUES (%s, %s)",
                [str(get_embedding(chunk)), chunk],
            )

        print(f"\nIndexed {len(chunks)} chunks from {filename}")
        db.commit()

    print(f"\nTotal index time: {time.perf_counter() - tic:.2f}s")

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