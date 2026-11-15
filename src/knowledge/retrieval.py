"""Curated business knowledge retrieval for the Retail Planner."""

import hashlib
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


BUSINESS_DOCUMENTS = (
    "model_card.md",
    "business_glossary.md",
    "forecast_methodology.md",
)

EXCLUDED_SECTIONS = {
    "Technical Evidence",
    "Provenance",
    "Sources and Scope",
}


def section_slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")

class BusinessKnowledgeRetriever:
    """Read-only, provenance-aware retrieval over curated documents."""

    def __init__(self, project_root):
        self.knowledge_dir = Path(project_root) / "docs" / "knowledge"
        self.passages = []
        self.source_hashes = {}

        self._load_passages()

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        self.matrix = self.vectorizer.fit_transform(
            [passage["content"] for passage in self.passages]
        )

    def _load_passages(self):
        for filename in BUSINESS_DOCUMENTS:
            path = self.knowledge_dir / filename
            content = path.read_text(encoding="utf-8")

            source_hash = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()

            self.source_hashes[filename] = source_hash

            match = re.search(
                r"\*\*Knowledge ID:\*\*\s*`([^`]+)`",
                content,
            )
            if not match:
                raise ValueError(f"Missing Knowledge ID: {filename}")

            knowledge_id = match.group(1)
            headings = list(re.finditer(r"(?m)^## (.+)$", content))

            for index, heading in enumerate(headings):
                title = heading.group(1).strip()

                if title in EXCLUDED_SECTIONS:
                    continue

                end = (
                    headings[index + 1].start()
                    if index + 1 < len(headings)
                    else len(content)
                )

                self.passages.append({
                    "passage_id": (
                        f"{knowledge_id}::{section_slug(title)}"
                    ),
                    "knowledge_id": knowledge_id,
                    "source_path": f"docs/knowledge/{filename}",
                    "source_sha256": source_hash,
                    "section": title,
                    "content": content[heading.start():end].strip(),
                })

        identifiers = [p["passage_id"] for p in self.passages]

        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("Empty corpus or duplicate passage identifiers.")

    def _verify_sources(self):
        for filename, expected_hash in self.source_hashes.items():
            path = self.knowledge_dir / filename
            current_hash = hashlib.sha256(path.read_bytes()).hexdigest()

            if current_hash != expected_hash:
                raise RuntimeError(
                    f"Knowledge source changed after indexing: {filename}"
                )

    def search(self, question, top_k=3):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a nonempty string.")

        if top_k < 1:
            raise ValueError("top_k must be positive.")

        self._verify_sources()

        query = self.vectorizer.transform([question])

        scores = (
            self.matrix @ query.T
        ).toarray().ravel()

        ranked = np.argsort(-scores, kind="stable")

        results = []

        for index in ranked[:top_k]:
            if scores[index] <= 0:
                continue

            results.append({
                **self.passages[index],
                "score": float(scores[index]),
            })

        return results

    def build_context(self, question, top_k=3):
        results = self.search(question, top_k=top_k)

        evidence = []

        for rank, result in enumerate(results, start=1):
            evidence.append({
                "rank": rank,
                **result,
            })

        return {
            "question": question,
            "retrieval_method": "curated_tfidf_v1",
            "evidence": evidence,
        }