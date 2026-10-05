"""
Shared text encoder used by classification (stage 2) and GraphRAG
retrieval (stage 3).

Primary backend:
LegalBERT (`nlpaueb/legal-bert-base-uncased`) using mean-pooled
domain-specific embeddings and attention weights for XAI.

Fallback backend:
A deterministic hashed bag-of-words encoder. It is used automatically
if LegalBERT cannot be loaded.

Important:
- `encode(text)` remains unchanged for existing code.
- `encode_batch(texts)` is added for fast FAISS/index construction.
- Existing classifier.py and legal_kb.py connections remain compatible.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np


MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

# Batch size used while generating embeddings for the knowledge base.
# 16 is a safe starting point for most CPU systems.
# If your PC has enough RAM and the process is stable, you can try 32.
DEFAULT_BATCH_SIZE = 16

# Maximum number of tokens processed from each clause.
MAX_LENGTH = 256


# Small stopword list shared by the BoW fallback vectorizer and the
# frequency-based XAI fallback in classifier.py.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "is", "are", "shall", "this", "that", "its", "be", "as", "any",
    "from", "under", "such", "which", "at", "if", "not", "all", "other",
    "will", "may", "than", "then", "into", "upon", "each", "either", "both",
}

_TOKEN_RE = re.compile(r"[A-Za-z']+")


# Singleton backend. It is created only when get_encoder() is first called.
_backend = None


# ============================================================
# LEGALBERT BACKEND
# ============================================================

class _LegalBertBackend:
    """
    LegalBERT backend.

    Supports:
        encode(text)
        encode_batch(texts)
        encode_with_attention(text)

    `encode()` is retained so the existing classifier and retrieval
    code do not need to change.
    """

    name = "legalbert"

    def __init__(self, tokenizer, model, torch_module):
        self._tok = tokenizer
        self._model = model
        self._torch = torch_module

        # Use GPU automatically if PyTorch detects one.
        # Otherwise, the model runs on CPU.
        self._device = (
            torch_module.device("cuda")
            if torch_module.cuda.is_available()
            else torch_module.device("cpu")
        )

        self._model.to(self._device)
        self._model.eval()

        print(f"[embeddings] LegalBERT device: {self._device}")

    def encode(self, text: str) -> np.ndarray:
        """
        Encode one text.

        This method keeps the old public interface unchanged.
        """
        vectors = self.encode_batch(
            [text],
            batch_size=1,
        )

        return vectors[0]

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode many texts efficiently.

        This is much faster than calling encode() once for every
        knowledge-base record because LegalBERT processes multiple
        clauses in one forward pass.

        Returns:
            NumPy array with shape:
                (number_of_texts, embedding_dimension)
        """

        if not texts:
            return np.empty(
                (0, self._model.config.hidden_size),
                dtype=np.float32,
            )

        torch = self._torch

        # Ensure every item is a string.
        clean_texts = [
            "" if text is None else str(text)
            for text in texts
        ]

        all_vectors: list[np.ndarray] = []

        total_batches = (
            len(clean_texts) + batch_size - 1
        ) // batch_size

        for batch_number, start in enumerate(
            range(0, len(clean_texts), batch_size),
            start=1,
        ):

            batch_texts = clean_texts[
                start:start + batch_size
            ]

            inputs = self._tok(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )

            # Move tokenizer output to CPU/GPU.
            inputs = {
                key: value.to(self._device)
                for key, value in inputs.items()
            }

            with torch.no_grad():

                out = self._model(
                    **inputs,
                    output_attentions=False,
                )

            # Mean pooling while ignoring padding tokens.
            mask = (
                inputs["attention_mask"]
                .unsqueeze(-1)
                .float()
            )

            summed = (
                out.last_hidden_state * mask
            ).sum(dim=1)

            counts = (
                mask.sum(dim=1)
                .clamp(min=1e-6)
            )

            vectors = summed / counts

            # L2 normalization makes inner-product FAISS search
            # equivalent to cosine-similarity search.
            vectors = torch.nn.functional.normalize(
                vectors,
                p=2,
                dim=1,
            )

            all_vectors.append(
                vectors.cpu().numpy().astype(
                    np.float32
                )
            )

            if show_progress:

                if (
                    batch_number == 1
                    or batch_number % 25 == 0
                    or batch_number == total_batches
                ):

                    completed = min(
                        start + len(batch_texts),
                        len(clean_texts),
                    )

                    percent = (
                        completed
                        / len(clean_texts)
                        * 100
                    )

                    print(
                        "[embeddings] "
                        f"Encoded {completed:,}"
                        f"/{len(clean_texts):,} "
                        f"records "
                        f"({percent:.1f}%)"
                    )

        return np.vstack(
            all_vectors
        ).astype(
            np.float32
        )

    def encode_with_attention(
        self,
        text: str,
    ) -> tuple[
        np.ndarray,
        list[tuple[str, float]],
    ]:
        """
        Returns:
            (
                embedding,
                [
                    (token, attention_from_CLS),
                    ...
                ]
            )

        This is kept for the classifier's attention-based explanation.
        """

        torch = self._torch

        inputs = self._tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            out = self._model(
                **inputs,
                output_attentions=True,
            )

        mask = (
            inputs["attention_mask"]
            .unsqueeze(-1)
            .float()
        )

        summed = (
            out.last_hidden_state * mask
        ).sum(dim=1)

        counts = (
            mask.sum(dim=1)
            .clamp(min=1e-6)
        )

        vector = summed / counts

        # Keep the embedding normalized just like batch embeddings.
        vector = torch.nn.functional.normalize(
            vector,
            p=2,
            dim=1,
        )

        embedding = (
            vector
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Last layer:
        # (batch, heads, sequence, sequence)
        last_layer_attention = (
            out.attentions[-1][0]
        )

        # Average all attention heads:
        # (sequence, sequence)
        average_attention = (
            last_layer_attention.mean(dim=0)
        )

        # Attention from the CLS token to every token:
        # (sequence,)
        cls_attention = (
            average_attention[0]
            .detach()
            .cpu()
            .tolist()
        )

        token_ids = (
            inputs["input_ids"][0]
            .detach()
            .cpu()
        )

        tokens = (
            self._tok
            .convert_ids_to_tokens(
                token_ids
            )
        )

        token_attention = list(
            zip(
                tokens,
                cls_attention,
            )
        )

        return (
            embedding,
            token_attention,
        )


# ============================================================
# BAG-OF-WORDS FALLBACK
# ============================================================

class _BowBackend:
    """
    Offline fallback backend.

    It exposes the same encode() and encode_batch() methods as
    LegalBERT, so the rest of the pipeline does not need to know
    which backend is active.
    """

    name = "bow_fallback"

    dim = 256

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:

        return [
            token.lower()
            for token in _TOKEN_RE.findall(
                text
            )
        ]

    def _hash_idx(
        self,
        token: str,
    ) -> int:

        digest = hashlib.md5(
            token.encode("utf-8")
        ).hexdigest()

        return int(
            digest,
            16,
        ) % self.dim

    def encode(
        self,
        text: str,
    ) -> np.ndarray:

        vector = np.zeros(
            self.dim,
            dtype=np.float32,
        )

        for token in self._tokenize(
            "" if text is None else str(text)
        ):

            if (
                token in STOPWORDS
                or len(token) < 2
            ):
                continue

            vector[
                self._hash_idx(token)
            ] += 1.0

        norm = np.linalg.norm(
            vector
        )

        if norm > 0:

            vector = vector / norm

        return vector.astype(
            np.float32
        )

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:

        # batch_size is accepted for interface compatibility.
        # The lightweight BoW encoder does not need torch batching.

        if not texts:

            return np.empty(
                (0, self.dim),
                dtype=np.float32,
            )

        vectors = []

        total = len(texts)

        for index, text in enumerate(
            texts,
            start=1,
        ):

            vectors.append(
                self.encode(text)
            )

            if show_progress:

                if (
                    index == 1
                    or index % 1000 == 0
                    or index == total
                ):

                    percent = (
                        index
                        / total
                        * 100
                    )

                    print(
                        "[embeddings] "
                        f"Encoded {index:,}"
                        f"/{total:,} "
                        f"records "
                        f"({percent:.1f}%)"
                    )

        return np.vstack(
            vectors
        ).astype(
            np.float32
        )


# ============================================================
# BACKEND LOADING
# ============================================================

def _try_load_legalbert() -> _LegalBertBackend | None:

    try:

        import torch

        from transformers import (
            AutoModel,
            AutoTokenizer,
        )

    except ImportError:

        return None

    try:

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                MODEL_NAME
            )
        )

        # Do not enable output_attentions globally.
        # Normal embedding generation does not need attention,
        # and disabling it makes bulk indexing faster.
        model = (
            AutoModel
            .from_pretrained(
                MODEL_NAME,
                attn_implementation="eager",
            )
        )

        model.eval()

    except Exception as error:

        print(
            "[embeddings] Could not load "
            f"{MODEL_NAME} "
            f"({error.__class__.__name__}: "
            f"{error})."
        )

        print(
            "[embeddings] Falling back to "
            "offline bag-of-words embeddings."
        )

        return None

    return _LegalBertBackend(
        tokenizer,
        model,
        torch,
    )


def get_encoder():
    """
    Return the shared encoder singleton.

    Existing code can continue using:

        encoder = get_encoder()
        vector = encoder.encode(text)

    New FAISS/index-building code can use:

        vectors = encoder.encode_batch(
            texts,
            batch_size=16,
            show_progress=True,
        )
    """

    global _backend

    if _backend is None:

        _backend = (
            _try_load_legalbert()
            or _BowBackend()
        )

        if _backend.name == "legalbert":

            print(
                "[embeddings] Using "
                f"{MODEL_NAME} "
                "for LegalBERT embeddings "
                "and attention-based XAI."
            )

        else:

            print(
                "[embeddings] Using the "
                "offline bag-of-words fallback."
            )

    return _backend


# ============================================================
# SIMILARITY
# ============================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a = np.asarray(
        a,
        dtype=np.float32,
    )

    b = np.asarray(
        b,
        dtype=np.float32,
    )

    norm_a = np.linalg.norm(a)

    norm_b = np.linalg.norm(b)

    if (
        norm_a == 0.0
        or norm_b == 0.0
    ):

        return 0.0

    return float(
        np.dot(a, b)
        / (norm_a * norm_b)
    )


# """Shared text encoder used by classification (stage 2) and GraphRAG
# retrieval (stage 3) — both are Veda's layer and both need the same notion of
# "how similar is this clause text to that other text".

# Primary backend: LegalBERT (`nlpaueb/legal-bert-base-uncased`) via
# `transformers`, giving real domain-tuned embeddings and real attention
# weights for XAI.

# Fallback backend: a small offline hashed bag-of-words vectorizer. It kicks in
# automatically if `torch`/`transformers` aren't installed, or if the pretrained
# weights can't be downloaded (no internet at runtime) — so the pipeline never
# crashes for a missing dependency, it just degrades from "LegalBERT embeddings"
# to "keyword-overlap embeddings" and logs which one is active. Swap-in is
# transparent to callers: both backends expose the same `.encode()` method.

# To get the real backend: `pip install torch transformers` (needs internet
# once, to download the ~440MB LegalBERT weights; they're cached afterwards).
# """
# from __future__ import annotations
# import hashlib
# import re
# import numpy as np

# MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

# # Small stopword list shared by the BoW fallback vectorizer and the
# # frequency-based XAI fallback in classifier.py.
# STOPWORDS = {
#     "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
#     "by", "is", "are", "shall", "this", "that", "its", "be", "as", "any",
#     "from", "under", "such", "which", "at", "if", "not", "all", "other",
#     "will", "may", "than", "then", "into", "upon", "each", "either", "both",
# }

# _TOKEN_RE = re.compile(r"[A-Za-z']+")

# _backend = None  # singleton, set on first get_encoder() call


# class _LegalBertBackend:
#     """Real backend: LegalBERT mean-pooled embeddings + last-layer attention."""

#     name = "legalbert"

#     def __init__(self, tokenizer, model, torch_module):
#         self._tok = tokenizer
#         self._model = model
#         self._torch = torch_module

#     def encode(self, text: str) -> np.ndarray:
#         vec, _ = self._forward(text)
#         return vec

#     def encode_with_attention(self, text: str) -> tuple[np.ndarray, list[tuple[str, float]]]:
#         """Returns (embedding, [(token, attention_from_CLS), ...])."""
#         return self._forward(text, need_attention=True)

#     def _forward(self, text: str, need_attention: bool = False):
#         torch = self._torch
#         inputs = self._tok(text, return_tensors="pt", truncation=True, max_length=256)
#         with torch.no_grad():
#             out = self._model(**inputs)

#         mask = inputs["attention_mask"].unsqueeze(-1).float()
#         summed = (out.last_hidden_state * mask).sum(1)
#         counts = mask.sum(1).clamp(min=1e-6)
#         vec = (summed / counts).squeeze(0).numpy()

#         pairs: list[tuple[str, float]] = []
#         if need_attention:
#             last_layer_attn = out.attentions[-1][0]        # (heads, seq, seq)
#             cls_attn = last_layer_attn.mean(0)[0]           # (seq,) CLS -> each token
#             tokens = self._tok.convert_ids_to_tokens(inputs["input_ids"][0])
#             pairs = list(zip(tokens, cls_attn.tolist()))
#         return vec, pairs


# class _BowBackend:
#     """Fallback backend: deterministic hashed bag-of-words, L2-normalized.
#     No downloads, no GPU, works anywhere Python + numpy runs."""

#     name = "bow_fallback"
#     dim = 256

#     def _tokenize(self, text: str) -> list[str]:
#         return [t.lower() for t in _TOKEN_RE.findall(text)]

#     def _hash_idx(self, token: str) -> int:
#         # md5 (not builtin hash()) so bucket assignment is stable across runs.
#         digest = hashlib.md5(token.encode("utf-8")).hexdigest()
#         return int(digest, 16) % self.dim

#     def encode(self, text: str) -> np.ndarray:
#         vec = np.zeros(self.dim, dtype=np.float64)
#         for tok in self._tokenize(text):
#             if tok in STOPWORDS or len(tok) < 2:
#                 continue
#             vec[self._hash_idx(tok)] += 1.0
#         norm = np.linalg.norm(vec)
#         return vec / norm if norm > 0 else vec


# def _try_load_legalbert() -> _LegalBertBackend | None:
#     try:
#         import torch
#         from transformers import AutoTokenizer, AutoModel
#     except ImportError:
#         return None
#     try:
#         tok = AutoTokenizer.from_pretrained(MODEL_NAME)
#         model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True)
#         model.eval()
#     except Exception as exc:  # offline / HF hub unreachable / bad cache
#         print(f"[embeddings] Could not load {MODEL_NAME} ({exc.__class__.__name__}: {exc}). "
#               f"Falling back to offline bag-of-words embeddings.")
#         return None
#     return _LegalBertBackend(tok, model, torch)


# def get_encoder():
#     """Lazy singleton: real LegalBERT if available, else the offline fallback."""
#     global _backend
#     if _backend is None:
#         _backend = _try_load_legalbert() or _BowBackend()
#         if _backend.name == "legalbert":
#             print(f"[embeddings] Using {MODEL_NAME} for embeddings + attention.")
#         else:
#             print("[embeddings] transformers/torch not available — using the offline "
#                   "bag-of-words fallback. `pip install torch transformers` (with internet "
#                   "access once) to enable real LegalBERT embeddings and attention-based XAI.")
#     return _backend


# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     a = np.asarray(a, dtype=np.float64)
#     b = np.asarray(b, dtype=np.float64)
#     na, nb = np.linalg.norm(a), np.linalg.norm(b)
#     if na == 0.0 or nb == 0.0:
#         return 0.0
#     return float(np.dot(a, b) / (na * nb))
