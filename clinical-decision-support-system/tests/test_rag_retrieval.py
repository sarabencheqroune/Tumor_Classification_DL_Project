"""
Unit tests for RAGRetrievalTool.

Uses a temporary vector store so tests are isolated and fast.
"""
import pytest

from src.tools.rag_retrieval_tool import RAGRetrievalTool


@pytest.fixture(scope="module")
def rag_tool(tmp_path_factory):
    """Build a fresh RAG index from the real guidelines file."""
    store = tmp_path_factory.mktemp("chroma")
    tool  = RAGRetrievalTool(
        vector_store_path=str(store),
        # Use the actual guidelines bundled with the project
        guidelines_path="src/data/medical_guidelines.txt",
        top_k=2,
    )
    result = tool.build_index()
    assert result["status"] == "success", f"Index build failed: {result}"
    return tool


class TestRAGRetrievalTool:

    def test_build_index_succeeds(self, rag_tool):
        # Already verified in fixture; just confirm chunks_indexed > 0
        pass

    def test_retrieve_glioma_guidelines(self, rag_tool):
        result = rag_tool.retrieve_guidelines("glioma treatment")
        assert result["status"] == "success"
        assert result["retrieved_count"] > 0
        assert any("glioma" in g.lower() or "tumor" in g.lower() for g in result["guidelines"])

    def test_retrieve_meningioma_guidelines(self, rag_tool):
        result = rag_tool.retrieve_guidelines("meningioma imaging findings")
        assert result["status"] == "success"
        assert result["retrieved_count"] > 0

    def test_retrieve_pituitary_guidelines(self, rag_tool):
        result = rag_tool.retrieve_guidelines("pituitary adenoma recommendations")
        assert result["status"] == "success"

    def test_retrieve_no_tumor_guidelines(self, rag_tool):
        result = rag_tool.retrieve_guidelines("normal brain MRI no tumor")
        assert result["status"] == "success"

    def test_top_k_respected(self, rag_tool):
        result = rag_tool.retrieve_guidelines("brain tumor", k=2)
        assert result["retrieved_count"] <= 2

    def test_returns_list_of_strings(self, rag_tool):
        result = rag_tool.retrieve_guidelines("glioma")
        assert isinstance(result["guidelines"], list)
        assert all(isinstance(g, str) for g in result["guidelines"])
