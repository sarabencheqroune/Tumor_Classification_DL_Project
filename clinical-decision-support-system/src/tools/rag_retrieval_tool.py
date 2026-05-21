"""
RAG Retrieval Tool — fetches relevant clinical guidelines from ChromaDB
given a tumor type or free-text query.

On first use, call build_index() to populate the vector store.
Subsequent uses call retrieve_guidelines() against the persisted index.
"""
import hashlib
from pathlib import Path
from typing import Optional

from src.config import EMBEDDING_MODEL, GUIDELINES_PATH, RAG_TOP_K, VECTOR_STORE_PATH


class RAGRetrievalTool:
    """
    ChromaDB-backed retrieval of medical guidelines.

    Lazy-initialises the embedding model and ChromaDB client on first use
    to avoid slowing down import time.
    """

    COLLECTION_NAME = "medical_guidelines"

    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        guidelines_path: Optional[str] = None,
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = RAG_TOP_K,
    ):
        self.vector_store_path = Path(vector_store_path or VECTOR_STORE_PATH)
        self.guidelines_path   = Path(guidelines_path   or GUIDELINES_PATH)
        self.embedding_model   = embedding_model
        self.top_k             = top_k

        self._client     = None
        self._collection = None
        self._embedder   = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def build_index(self) -> dict:
        """
        Chunk the guidelines file and (re)build the ChromaDB vector store.
        Call once during setup or when guidelines are updated.
        """
        self._ensure_embedder()
        client     = self._get_client()
        # Drop and recreate for idempotency
        try:
            client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        collection = client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        if not self.guidelines_path.exists():
            return {"status": "error", "message": f"Guidelines file not found: {self.guidelines_path}"}

        text = self.guidelines_path.read_text(encoding="utf-8")
        chunks = self._split_into_sections(text)

        ids, embeddings, documents, metadatas = [], [], [], []
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            ids.append(chunk_id)
            embeddings.append(self._embedder.encode(chunk).tolist())
            documents.append(chunk)
            metadatas.append({"source": str(self.guidelines_path)})

        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        self._collection = collection

        return {"status": "success", "chunks_indexed": len(chunks)}

    def retrieve_guidelines(self, query: str, k: Optional[int] = None) -> dict:
        """
        Retrieve the top-k most relevant guideline sections for a query.

        Args:
            query:  e.g. "glioma treatment recommendations"
            k:      override default top_k

        Returns:
            {
                "status": "success",
                "query": "...",
                "guidelines": ["text1", "text2", ...],
                "retrieved_count": 3
            }
        """
        k = k or self.top_k
        try:
            self._ensure_embedder()
            self._ensure_collection()

            q_embedding = self._embedder.encode(query).tolist()
            results = self._collection.query(
                query_embeddings=[q_embedding],
                n_results=k,
            )
            docs = results.get("documents", [[]])[0]

            return {
                "status": "success",
                "query": query,
                "guidelines": docs,
                "retrieved_count": len(docs),
            }

        except Exception as exc:
            return {
                "status": "error",
                "message": str(exc),
                "guidelines": [],
                "retrieved_count": 0,
            }

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _get_client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.vector_store_path))
        return self._client

    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model)

    def _ensure_collection(self):
        if self._collection is None:
            client = self._get_client()
            try:
                self._collection = client.get_collection(self.COLLECTION_NAME)
            except Exception:
                # Index hasn't been built yet — build it automatically
                result = self.build_index()
                if result["status"] != "success":
                    raise RuntimeError(f"Auto-index build failed: {result.get('message')}")
                self._collection = client.get_collection(self.COLLECTION_NAME)

    @staticmethod
    def _split_into_sections(text: str) -> list[str]:
        """Split on '---' separators and remove empty chunks."""
        sections = [s.strip() for s in text.split("---")]
        return [s for s in sections if len(s) > 50]
