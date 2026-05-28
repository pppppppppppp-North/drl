from abc import ABC, abstractmethod
import torch
from typing import List

class BaseLLMExtractor(ABC):
    """
    Abstract base class for LLM feature extractors.
    """
    @abstractmethod
    def get_embedding(self, text_list: List[str]) -> torch.Tensor:
        """
        Converts a list of text strings into a tensor of embeddings.
        Output shape: (batch_size, embedding_dim)
        """
        pass

class MockLLMExtractor(BaseLLMExtractor):
    """
    Mock extractor for testing/development without a heavy LLM.
    Returns random noise vectors.
    """
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def get_embedding(self, text_list: List[str]) -> torch.Tensor:
        batch_size = len(text_list)
        # Simulate processing time or logic here if needed
        return torch.randn(batch_size, self.embedding_dim)

# Future: class HuggingFaceLLMExtractor(BaseLLMExtractor): ...
# Future: class OpenAIEmbeddingExtractor(BaseLLMExtractor): ...

if __name__ == "__main__":
    extractor = MockLLMExtractor()
    texts = ["Market is bullish today", "Inflation concerns rise"]
    embeddings = extractor.get_embedding(texts)
    print(f"Embeddings shape: {embeddings.shape}")
