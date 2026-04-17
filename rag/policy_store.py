"""
RAG policy store using FAISS + sentence-transformers.

Falls back to keyword-based search if faiss-cpu is not installed.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports – graceful degradation
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed – falling back to keyword search.")

try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed – keyword search only.")


# ---------------------------------------------------------------------------
# PolicyStore
# ---------------------------------------------------------------------------

class PolicyStore:
    """Manages insurance policy documents for semantic retrieval."""

    def __init__(self, embed_model: str = "all-MiniLM-L6-v2", store_path: str = "./data/vector_store"):
        self.embed_model_name = embed_model
        self.store_path = store_path
        self.encoder: Optional[Any] = None
        self.index: Optional[Any] = None  # faiss index
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

        os.makedirs(store_path, exist_ok=True)
        self._load_encoder()
        self.load_or_create()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_encoder(self) -> None:
        if ST_AVAILABLE:
            try:
                self.encoder = SentenceTransformer(self.embed_model_name)
                logger.info("SentenceTransformer loaded: %s", self.embed_model_name)
            except Exception as exc:
                logger.error("Failed to load SentenceTransformer: %s", exc)
                self.encoder = None

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised embeddings as float32 numpy array."""
        if self.encoder is None:
            raise RuntimeError("No embedding encoder available.")
        vecs = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vecs = vecs.astype(np.float32)
        faiss.normalize_L2(vecs)
        return vecs

    def _index_path(self) -> str:
        return os.path.join(self.store_path, "policy.index")

    def _meta_path(self) -> str:
        return os.path.join(self.store_path, "policy_meta.pkl")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_or_create(self) -> None:
        """Load persisted FAISS index or build a fresh one with sample data."""
        idx_path = self._index_path()
        meta_path = self._meta_path()

        if FAISS_AVAILABLE and ST_AVAILABLE and os.path.exists(idx_path) and os.path.exists(meta_path):
            try:
                self.index = faiss.read_index(idx_path)
                with open(meta_path, "rb") as fh:
                    saved = pickle.load(fh)
                self.documents = saved["documents"]
                self.metadata = saved["metadata"]
                logger.info("Loaded existing policy FAISS index (%d docs).", len(self.documents))
                return
            except Exception as exc:
                logger.warning("Could not load existing index: %s – rebuilding.", exc)

        # Build from scratch
        sample_policies = self.get_sample_policies()
        for item in sample_policies:
            self.add_policy(item["text"], item["metadata"])
        logger.info("Policy store initialised with %d sample documents.", len(self.documents))

    def add_policy(self, policy_text: str, metadata: Dict[str, Any]) -> None:
        """Encode and add a policy document to the store."""
        self.documents.append(policy_text)
        self.metadata.append(metadata)

        if FAISS_AVAILABLE and ST_AVAILABLE and self.encoder is not None:
            vec = self._encode([policy_text])  # shape (1, dim)
            if self.index is None:
                dim = vec.shape[1]
                self.index = faiss.IndexFlatIP(dim)  # inner-product on normalised vecs = cosine
            self.index.add(vec)

            # Persist
            try:
                faiss.write_index(self.index, self._index_path())
                with open(self._meta_path(), "wb") as fh:
                    pickle.dump({"documents": self.documents, "metadata": self.metadata}, fh)
            except Exception as exc:
                logger.warning("Could not persist policy index: %s", exc)

    def retrieve_policy(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Return up to *k* relevant policy chunks for *query*.

        Each result is a dict with keys: ``text``, ``metadata``, ``score``.
        Falls back to keyword overlap when FAISS/ST unavailable.
        """
        if not self.documents:
            return []

        k = min(k, len(self.documents))

        # --- FAISS semantic search ---
        if FAISS_AVAILABLE and ST_AVAILABLE and self.index is not None and self.encoder is not None:
            try:
                vec = self._encode([query])
                scores, indices = self.index.search(vec, k)
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0:
                        continue
                    results.append({
                        "text": self.documents[idx],
                        "metadata": self.metadata[idx],
                        "score": float(score),
                    })
                return results
            except Exception as exc:
                logger.warning("FAISS search failed: %s – falling back to keyword.", exc)

        # --- Keyword fallback ---
        query_tokens = set(query.lower().split())
        scored = []
        for i, doc in enumerate(self.documents):
            doc_tokens = set(doc.lower().split())
            overlap = len(query_tokens & doc_tokens)
            scored.append((overlap, i))
        scored.sort(reverse=True)
        return [
            {"text": self.documents[i], "metadata": self.metadata[i], "score": float(s)}
            for s, i in scored[:k]
        ]

    # ------------------------------------------------------------------
    # Sample data
    # ------------------------------------------------------------------

    @staticmethod
    def get_sample_policies() -> List[Dict[str, Any]]:
        """Return a list of sample insurance policy documents."""
        return [
            {
                "text": (
                    "Policy INS-2024-001 — Holder: John Smith (john.smith@email.com). "
                    "Coverage types: fire damage, water damage, collision. "
                    "Deductible: $500. Maximum coverage: $10,000. "
                    "Exclusions: intentional damage, war, nuclear events. "
                    "This policy covers accidental fire damage to the insured vehicle or property, "
                    "water damage from flooding or burst pipes, and collision damage from road accidents. "
                    "Claims must be filed within 30 days of the incident. "
                    "Proof of incident (police report or fire brigade report) is required."
                ),
                "metadata": {
                    "policy_number": "INS-2024-001",
                    "customer_name": "John Smith",
                    "customer_email": "john.smith@email.com",
                    "coverage_types": ["fire", "water damage", "collision"],
                    "deductible": 500,
                    "max_coverage": 10000,
                    "exclusions": ["intentional damage", "war", "nuclear events"],
                },
            },
            {
                "text": (
                    "Policy INS-2024-002 — Holder: Jane Doe (jane.doe@email.com). "
                    "Coverage types: collision, theft. "
                    "Deductible: $1,000. Maximum coverage: $25,000. "
                    "Exclusions: racing, DUI incidents, cosmetic damage. "
                    "This policy provides comprehensive collision coverage for road accidents and "
                    "theft protection for the insured vehicle. "
                    "For theft claims, a police report number is mandatory. "
                    "Collision claims require photographs of the damage and a repair estimate "
                    "from a certified mechanic within 14 days."
                ),
                "metadata": {
                    "policy_number": "INS-2024-002",
                    "customer_name": "Jane Doe",
                    "customer_email": "jane.doe@email.com",
                    "coverage_types": ["collision", "theft"],
                    "deductible": 1000,
                    "max_coverage": 25000,
                    "exclusions": ["racing", "DUI incidents", "cosmetic damage"],
                },
            },
            {
                "text": (
                    "Policy INS-2024-003 — Holder: Bob Johnson (bob.johnson@email.com). "
                    "Coverage types: comprehensive (fire, water damage, collision, theft, "
                    "natural disaster, vandalism). "
                    "Deductible: $250. Maximum coverage: $50,000. "
                    "Exclusions: mechanical breakdown, normal wear and tear, consequential losses. "
                    "This is a comprehensive all-risk policy offering the broadest coverage available. "
                    "Natural disaster coverage includes hail, flood, earthquake, and tornado damage. "
                    "Vandalism claims require a police report. "
                    "All claims are subject to independent assessment by an approved loss adjuster."
                ),
                "metadata": {
                    "policy_number": "INS-2024-003",
                    "customer_name": "Bob Johnson",
                    "customer_email": "bob.johnson@email.com",
                    "coverage_types": [
                        "fire", "water damage", "collision", "theft",
                        "natural disaster", "vandalism",
                    ],
                    "deductible": 250,
                    "max_coverage": 50000,
                    "exclusions": [
                        "mechanical breakdown", "normal wear and tear", "consequential losses"
                    ],
                },
            },
            # Extra granular chunks for better retrieval
            {
                "text": (
                    "Collision coverage applies when the insured vehicle is damaged due to a road "
                    "accident involving another vehicle or stationary object. "
                    "The insured must not be under the influence of alcohol or narcotics. "
                    "The claim amount is calculated as repair cost minus the applicable deductible."
                ),
                "metadata": {"coverage_type": "collision", "chunk_type": "general_terms"},
            },
            {
                "text": (
                    "Fire damage coverage applies when the insured asset is damaged by accidental fire. "
                    "Deliberate fire or arson voids the policy. "
                    "Coverage includes smoke damage directly caused by the fire event."
                ),
                "metadata": {"coverage_type": "fire", "chunk_type": "general_terms"},
            },
            {
                "text": (
                    "Water damage coverage applies to flooding, burst pipes, and storm surges. "
                    "Gradual leakage or seepage not caused by a sudden event is excluded. "
                    "The insured must take reasonable steps to mitigate further damage."
                ),
                "metadata": {"coverage_type": "water damage", "chunk_type": "general_terms"},
            },
        ]
