"""Load models once and run four-way Java summarization comparisons."""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

from engine.models import CodeT5Model, LexRankModel, SentenceTransformerModel, TFIDFModel
from engine.preprocessing import split_code_statements, split_java_methods

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
IDF_CACHE = CACHE_DIR / "idf_weights_train_val.pkl"
DATASET = "google/code_x_glue_ct_code_to_text"
FIT_SPLITS = ("train", "validation")


@dataclass
class MethodSummary:
    name: str
    summary: str


@dataclass
class ModelSummary:
    model_id: str
    model: str
    tier: str
    approach: str
    accent: str
    summary: str
    elapsed_ms: float
    methods: list[MethodSummary]


@dataclass
class ComparisonResult:
    filename: str
    char_count: int
    token_count: int
    statement_count: int
    method_count: int
    top_n: int
    summaries: list[ModelSummary]
    total_elapsed_ms: float


MODEL_CATALOG = [
    {
        "id": "tfidf",
        "name": "TF-IDF",
        "glyph": "TF",
        "accent": "#2dd4bf",
        "family": "Statistical · Bag-of-words",
        "tier": "Corpus-fitted extractive",
        "approach": "Extractive",
        "input": "Statement fragments",
        "checkpoint": None,
        "description": "Scores each code statement by TF-IDF using IDF weights fitted on the CodeXGLUE Java train + validation corpus.",
        "tagline": "Picks statements packed with rare, high-signal terms.",
        "steps": [
            "Fit inverse-document-frequency (IDF) weights over the Java train + validation corpus.",
            "Tokenize each statement: split identifiers, lowercase, drop stopwords.",
            "Score every statement as the sum of term-frequency x IDF.",
            "Return the top-N highest-scoring statements as the summary.",
        ],
        "strengths": ["Fast and fully offline", "Interpretable scores", "No GPU required"],
        "limitations": ["Output is code-like, not prose", "Ignores word order and context"],
        "speed": "Instant",
    },
    {
        "id": "lexrank",
        "name": "LexRank",
        "glyph": "LR",
        "accent": "#38bdf8",
        "family": "Graph · Centrality",
        "tier": "Corpus-fitted extractive",
        "approach": "Extractive",
        "input": "Statement fragments",
        "checkpoint": None,
        "description": "Builds a similarity graph over statements and runs PageRank to pick the most central fragments.",
        "tagline": "Selects statements most representative of the whole file.",
        "steps": [
            "Build a TF-IDF vector for each statement using shared corpus IDF.",
            "Compute pairwise cosine similarity to form a statement graph.",
            "Threshold weak edges, then run PageRank over the graph.",
            "Return the most central (highest-ranked) statements.",
        ],
        "strengths": ["Captures redundancy / centrality", "Offline and interpretable", "Robust on longer files"],
        "limitations": ["Needs several statements to rank", "Still extractive, not generative"],
        "speed": "Fast",
    },
    {
        "id": "sentence_transformers",
        "name": "SentenceTransformers",
        "glyph": "ST",
        "accent": "#a78bfa",
        "family": "Neural · Sentence embeddings",
        "tier": "General-language pretrained",
        "approach": "Extractive",
        "input": "Statement fragments",
        "checkpoint": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Encodes statements with all-MiniLM-L6-v2 and selects those closest to the centroid embedding.",
        "tagline": "Uses semantic meaning to find the most central statements.",
        "steps": [
            "Embed each statement with the all-MiniLM-L6-v2 transformer.",
            "Average the embeddings into a single centroid vector.",
            "Rank statements by cosine similarity to the centroid.",
            "Return the statements closest to the semantic center.",
        ],
        "strengths": ["Understands English semantics", "Order-aware encoder", "No corpus fitting needed"],
        "limitations": ["Pretrained on prose, not code", "Heavier than TF-IDF/LexRank"],
        "speed": "Moderate",
    },
    {
        "id": "codet5",
        "name": "CodeT5",
        "glyph": "T5",
        "accent": "#f59e0b",
        "family": "Transformer · Seq2seq",
        "tier": "Code-specific fine-tuned",
        "approach": "Abstractive",
        "input": "Per-method Java source (256-token window each)",
        "checkpoint": "Salesforce/codet5-base-codexglue-sum-java",
        "description": "Generates natural-language summaries from raw Java source using a CodeT5 checkpoint fine-tuned on CodeXGLUE.",
        "tagline": "Writes a fresh English sentence describing the code.",
        "steps": [
            "Split the file into individual Java methods.",
            "Byte-level BPE tokenize each method (first 256 tokens).",
            "Decode with beam search — one English sentence per method, same as evaluation.",
            "Show each method summary separately in the results view.",
        ],
        "strengths": ["True natural-language output", "Fine-tuned on Java code-comment pairs", "Best quality summaries"],
        "limitations": ["Slow on CPU", "256-token input limit", "Can hallucinate details"],
        "speed": "Slowest",
    },
]


class SummarizationPipeline:
    def __init__(self, top_n: int = 5) -> None:
        self.top_n = top_n
        self.ready = False
        self.loading = False
        self.error: str | None = None
        self.tfidf: TFIDFModel | None = None
        self.lexrank: LexRankModel | None = None
        self.st_model: SentenceTransformerModel | None = None
        self.codet5: CodeT5Model | None = None

    def load(self) -> None:
        if self.ready or self.loading:
            return
        self.loading = True
        self.error = None
        try:
            idf, n = self._load_or_build_idf()
            logger.info("Fitting TF-IDF and LexRank from cached IDF (%d terms)", len(idf))
            self.tfidf = TFIDFModel().load_idf(idf, n)
            self.lexrank = LexRankModel().load_idf(idf)

            logger.info("Loading SentenceTransformers ...")
            self.st_model = SentenceTransformerModel()

            logger.info("Loading CodeT5 ...")
            self.codet5 = CodeT5Model()

            self.ready = True
            logger.info("Pipeline ready.")
        except Exception as exc:
            self.error = str(exc)
            logger.exception("Failed to load summarization pipeline")
            raise
        finally:
            self.loading = False

    def _load_or_build_idf(self) -> tuple[dict[str, float], int]:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if IDF_CACHE.exists():
            logger.info("Loading IDF cache from %s", IDF_CACHE)
            with IDF_CACHE.open("rb") as f:
                payload = pickle.load(f)
            return payload["idf"], payload["N"]

        logger.info("Building IDF from %s (%s) ...", DATASET, " + ".join(FIT_SPLITS))
        dataset = load_dataset(DATASET, "java")
        fit_data = concatenate_datasets([dataset[split] for split in FIT_SPLITS])
        fit_corpus: list[str] = []
        for row in fit_data:
            fit_corpus.extend(split_code_statements(row["code"]))

        tfidf = TFIDFModel().fit(fit_corpus)
        with IDF_CACHE.open("wb") as f:
            pickle.dump({"idf": tfidf.idf, "N": tfidf.N}, f)
        logger.info(
            "Cached IDF weights (%d terms, %d statements from %d methods)",
            len(tfidf.idf),
            len(fit_corpus),
            len(fit_data),
        )
        return tfidf.idf, tfidf.N

    def compare(self, java_source: str, filename: str = "upload.java") -> ComparisonResult:
        if not self.ready:
            raise RuntimeError("Pipeline is not ready. Call load() first.")

        source = java_source.strip()
        if not source:
            raise ValueError("Java source is empty.")

        statements = split_code_statements(source)
        java_methods = split_java_methods(source)
        summaries: list[ModelSummary] = []
        t_total = time.perf_counter()
        catalog_by_id = {m["id"]: m for m in MODEL_CATALOG}

        extractive_runners = [
            ("tfidf", lambda: " ".join(self.tfidf.summarize(statements, self.top_n))),
            ("lexrank", lambda: " ".join(self.lexrank.summarize(statements, self.top_n))),
            (
                "sentence_transformers",
                lambda: " ".join(self.st_model.summarize(statements, self.top_n)),
            ),
        ]

        for model_id, run in extractive_runners:
            meta = catalog_by_id[model_id]
            t0 = time.perf_counter()
            text = run()
            summaries.append(ModelSummary(
                model_id=model_id,
                model=meta["name"],
                tier=meta["tier"],
                approach=meta["approach"],
                accent=meta["accent"],
                summary=text,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                methods=[],
            ))

        codet5_meta = catalog_by_id["codet5"]
        t0 = time.perf_counter()
        codet5_methods: list[MethodSummary] = []
        for method in java_methods:
            codet5_methods.append(MethodSummary(
                name=method["name"],
                summary=self.codet5.summarize(method["code"]),
            ))
        codet5_combined = "\n".join(
            m.summary.strip() for m in codet5_methods if m.summary.strip()
        )
        summaries.append(ModelSummary(
            model_id="codet5",
            model=codet5_meta["name"],
            tier=codet5_meta["tier"],
            approach=codet5_meta["approach"],
            accent=codet5_meta["accent"],
            summary=codet5_combined,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            methods=codet5_methods,
        ))

        return ComparisonResult(
            filename=filename,
            char_count=len(source),
            token_count=len(source.split()),
            statement_count=len(statements),
            method_count=len(java_methods),
            top_n=self.top_n,
            summaries=summaries,
            total_elapsed_ms=(time.perf_counter() - t_total) * 1000,
        )
