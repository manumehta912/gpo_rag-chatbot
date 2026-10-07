# GPO 2018 Annual Report - Baseline RAG Chatbot

Pipeline: PDF -> page text (pypdf) -> recursive chunking (1000 chars, 150 overlap, page metadata kept)
-> Gemini embeddings (`gemini-embedding-001`, 768-d, saved to `index/`) -> cosine top-5 retrieval
-> grounded answer with Gemini (`gemini-2.5-flash`) -> FastAPI + web chat UI.

## Setup
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # then paste your GEMINI_API_KEY
python ingest.py                                     # builds the index (once)
uvicorn app:app --reload                             # open http://127.0.0.1:8000
```

## Validate with the sample questions
Paste the assignment's sample questions into `questions.txt` (one per line), then:
```bash
python evaluate.py     # prints answers + retrieved pages, writes results.md
```
Check each answer against the PDF page shown in the sources.

## Tuning knobs (config.py)
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, EMBED_DIM. Model names are set in `.env`.
