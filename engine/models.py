

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoModelForSeq2SeqLM

from engine.preprocessing import tokenize


class TFIDFModel:

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self.N = 0

    def fit(self, corpus: list[str]) -> TFIDFModel:
        n = len(corpus)
        df: dict[str, int] = defaultdict(int)
        for sent in corpus:
            for term in set(tokenize(sent)):
                df[term] += 1
        self.idf = {
            term: math.log((n + 1) / (freq + 1)) + 1
            for term, freq in df.items()
        }
        self.N = n
        return self

    def load_idf(self, idf: dict[str, float], n: int) -> TFIDFModel:
        self.idf = idf
        self.N = n
        return self

    def _score(self, sentence: str) -> float:
        tokens = tokenize(sentence)
        if not tokens:
            return 0.0
        tf = Counter(tokens)
        return sum(tf[t] / len(tokens) * self.idf.get(t, 1.0) for t in tf)

    def summarize(self, sentences: list[str], top_n: int = 1) -> list[str]:
        if not sentences:
            return [""]
        scored = sorted(sentences, key=self._score, reverse=True)
        return scored[:top_n]


class LexRankModel:

    THRESHOLD = 0.1
    DAMPING = 0.85
    MAX_ITER = 100
    TOL = 1e-6

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}

    def fit(self, corpus: list[str]) -> LexRankModel:
        n = len(corpus)
        df: dict[str, int] = defaultdict(int)
        for sent in corpus:
            for term in set(tokenize(sent)):
                df[term] += 1
        self.idf = {
            term: math.log((n + 1) / (freq + 1)) + 1
            for term, freq in df.items()
        }
        return self

    def load_idf(self, idf: dict[str, float]) -> LexRankModel:
        self.idf = idf
        return self

    def _tfidf_vec(self, sentence: str) -> dict[str, float]:
        tokens = tokenize(sentence)
        if not tokens:
            return {}
        tf = Counter(tokens)
        return {t: (tf[t] / len(tokens)) * self.idf.get(t, 1.0) for t in tf}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v ** 2 for v in a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _pagerank(self, matrix: np.ndarray) -> np.ndarray:
        n = len(matrix)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        p = matrix / row_sums
        scores = np.ones(n) / n
        for _ in range(self.MAX_ITER):
            new_scores = (1 - self.DAMPING) / n + self.DAMPING * p.T @ scores
            if np.abs(new_scores - scores).sum() < self.TOL:
                break
            scores = new_scores
        return scores

    def summarize(self, sentences: list[str], top_n: int = 1) -> list[str]:
        if len(sentences) == 1:
            return sentences[:top_n]
        vecs = [self._tfidf_vec(s) for s in sentences]
        n = len(sentences)
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                c = self._cosine(vecs[i], vecs[j])
                if c >= self.THRESHOLD:
                    sim[i, j] = sim[j, i] = c
        if sim.sum() == 0:
            scored = sorted(range(n), key=lambda i: sum(vecs[i].values()), reverse=True)
            return [sentences[i] for i in scored[:top_n]]
        scores = self._pagerank(sim)
        ranked = np.argsort(scores)[::-1]
        return [sentences[i] for i in ranked[:top_n]]


class SentenceTransformerModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def summarize(self, sentences: list[str], top_n: int = 1) -> list[str]:
        if not sentences:
            return [""]
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        centroid = embeddings.mean(axis=0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        sims = (embeddings / norms) @ (centroid / (np.linalg.norm(centroid) + 1e-9))
        ranked = np.argsort(sims)[::-1]
        return [sentences[i] for i in ranked[:top_n]]


class CodeT5Model:
    MODEL_NAME = "Salesforce/codet5-base-codexglue-sum-java"
    VOCAB_REPO = "Salesforce/codet5-base"
    _SPECIAL_TOKENS = ("<pad>", "<s>", "</s>", "<unk>", "<mask>")

    def __init__(self) -> None:
        vocab_file = hf_hub_download(self.VOCAB_REPO, "vocab.json")
        merges_file = hf_hub_download(self.VOCAB_REPO, "merges.txt")
        self.tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def _clean(self, text: str) -> str:
        for tok in self._SPECIAL_TOKENS:
            text = text.replace(tok, " ")
        text = re.sub(r"<extra_id_\d+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def summarize(self, raw_code: str) -> str:
        if not raw_code or not raw_code.strip():
            return ""

        ids = self.tokenizer.encode(raw_code).ids[:256]
        input_ids = torch.tensor([ids], device=self.device)
        attention = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=48,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        decoded = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)
        return self._clean(decoded)
