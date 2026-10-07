"""Retrieval + grounded generation."""
import json

import numpy as np
from google import genai
from google.genai import types

import config

SYSTEM_PROMPT = """You are an assistant answering questions about the U.S. Government Publishing Office (GPO) 2018 Annual Report.
Answer ONLY using the numbered context excerpts provided.
- If the answer is not in the excerpts, say: "I couldn't find that in the GPO 2018 Annual Report."
- Do not use outside knowledge and do not guess numbers, names or dates.
- Be concise and accurate. Quote figures exactly as written.
- Cite the pages you used at the end like: (Source: p. 12, p. 15)."""


class RAG:
    def __init__(self):
        if not config.API_KEY:
            raise RuntimeError("GEMINI_API_KEY missing. Create a .env file (see .env.example).")
        self.client = genai.Client(api_key=config.API_KEY)
        self.vectors = np.load(config.INDEX_DIR / "embeddings.npy")
        self.chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text())

    def embed_query(self, q):
        res = self.client.models.embed_content(
            model=config.EMBED_MODEL,
            contents=[q],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY", output_dimensionality=config.EMBED_DIM
            ),
        )
        v = np.array(res.embeddings[0].values, dtype="float32")
        return v / np.linalg.norm(v)

    def retrieve(self, question, k=config.TOP_K):
        qv = self.embed_query(question)
        scores = self.vectors @ qv                     # cosine similarity
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def answer(self, question, history=None, k=config.TOP_K):
        hits = self.retrieve(question, k)
        context = "\n\n".join(
            f"[{n}] (page {h['page']})\n{h['text']}" for n, h in enumerate(hits, start=1)
        )
        convo = ""
        for turn in (history or [])[-4:]:
            convo += f"{turn['role'].capitalize()}: {turn['content']}\n"
        prompt = f"Context excerpts:\n{context}\n\n{convo}Question: {question}\nAnswer:"
        res = self.client.models.generate_content(
            model=config.CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=0.1
            ),
        )
        return {
            "answer": (res.text or "").strip(),
            "sources": [
                {"page": h["page"], "score": round(h["score"], 3), "text": h["text"][:300]}
                for h in hits
            ],
        }
