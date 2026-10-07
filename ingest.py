"""Ingestion: PDF -> pages -> chunks -> Gemini embeddings -> saved index.

Run once:  python ingest.py
"""
import json
import re
import sys
import time

import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader

import config


def load_pages(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)          # collapse spaces
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > 50:                            # skip blank/cover pages
            pages.append({"page": i, "text": text})
    return pages


def split_text(text, size, overlap):
    """Recursive splitter: paragraphs -> lines -> sentences -> words."""
    if len(text) <= size:
        return [text]
    for sep in ["\n\n", "\n", ". ", " "]:
        parts = text.split(sep)
        if len(parts) > 1:
            break
    chunks, current = [], ""
    for part in parts:
        piece = part + (sep if sep.strip() == "" or sep == ". " else sep)
        if len(piece) > size:                         # piece itself too big
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.extend(split_text(part, size, overlap))
            continue
        if len(current) + len(piece) > size and current:
            chunks.append(current.strip())
            current = current[-overlap:] + piece      # carry overlap forward
        else:
            current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_chunks(pages):
    chunks = []
    for p in pages:
        for j, c in enumerate(split_text(p["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            if len(c) < 40:
                continue
            chunks.append({"id": len(chunks), "page": p["page"], "part": j, "text": c})
    return chunks


def _retry_delay(err, default):
    """Pull 'Please retry in 33.6s' out of a 429 error message."""
    m = re.search(r"retry in ([\d.]+)s", str(err))
    return float(m.group(1)) + 2 if m else default


def embed_texts(client, texts, task_type, batch=50, pause=40):
    """Free tier allows ~100 embedding requests/min (each text counts as one),
    so we send 50 at a time and wait between batches. Progress is cached so a
    crash/re-run resumes instead of starting over."""
    config.INDEX_DIR.mkdir(exist_ok=True)
    cache = config.INDEX_DIR / "partial_embeddings.json"
    vecs = json.loads(cache.read_text()) if cache.exists() else []
    if vecs:
        print(f"  resuming from {len(vecs)}/{len(texts)}")

    for i in range(len(vecs), len(texts), batch):
        batch_texts = texts[i:i + batch]
        for attempt in range(10):
            try:
                res = client.models.embed_content(
                    model=config.EMBED_MODEL,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=config.EMBED_DIM,
                    ),
                )
                vecs.extend(e.values for e in res.embeddings)
                break
            except Exception as e:
                wait = _retry_delay(e, 2 ** attempt * 5)
                print(f"  rate limit/error; waiting {wait:.0f}s then retrying")
                time.sleep(wait)
        else:
            raise RuntimeError("Embedding failed after retries")
        cache.write_text(json.dumps(vecs))
        print(f"  embedded {len(vecs)}/{len(texts)}")
        if len(vecs) < len(texts):
            time.sleep(pause)

    arr = np.array(vecs, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)   # normalise (needed for truncated dims)
    cache.unlink(missing_ok=True)
    return arr


def main():
    if not config.API_KEY:
        sys.exit("GEMINI_API_KEY missing. Create a .env file (see .env.example).")
    client = genai.Client(api_key=config.API_KEY)

    print("Reading PDF...")
    pages = load_pages(config.PDF_PATH)
    chunks = build_chunks(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")

    print("Embedding chunks...")
    vectors = embed_texts(client, [c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")

    config.INDEX_DIR.mkdir(exist_ok=True)
    np.save(config.INDEX_DIR / "embeddings.npy", vectors)
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1))
    print(f"Saved index to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
