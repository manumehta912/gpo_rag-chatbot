"""Validate against the assignment's sample questions.

Put one question per line in questions.txt, then:  python evaluate.py
Writes results.md with answer + retrieved pages for manual checking.
"""
from pathlib import Path

import config
from rag import RAG

qfile = config.BASE_DIR / "questions.txt"
questions = [q.strip() for q in qfile.read_text().splitlines() if q.strip()]
rag = RAG()
out = ["# Evaluation results\n"]
for q in questions:
    r = rag.answer(q)
    pages = ", ".join(f"p.{s['page']}" for s in r["sources"])
    print(f"\nQ: {q}\nA: {r['answer']}\n   retrieved: {pages}")
    out.append(f"## {q}\n\n{r['answer']}\n\n*Retrieved pages: {pages}*\n")
Path(config.BASE_DIR / "results.md").write_text("\n".join(out))
print("\nSaved results.md")
