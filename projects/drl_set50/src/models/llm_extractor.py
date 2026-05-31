from abc import ABC, abstractmethod
import torch
from typing import List, Union
from transformers import AutoTokenizer, AutoModel

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
        return torch.randn(batch_size, self.embedding_dim)

class FinBERTExtractor(BaseLLMExtractor):
    """
    Uses FinBERT (ProsusAI) to extract financial context embeddings.
    """
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "cpu"):
        self.device = device
        print(f"Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_embedding(self, text_list: List[str]) -> torch.Tensor:
        """
        Returns the [CLS] token embedding for the input texts.
        """
        # Tokenize
        inputs = self.tokenizer(
            text_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the CLS token (first token) as the sentence embedding
            # Shape: (batch_size, hidden_dim=768)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings.cpu()

if __name__ == "__main__":
    # Test
    try:
        extractor = FinBERTExtractor()
        texts = ["The market is crashing due to inflation.", "Stocks hit all-time high."]
        embeddings = extractor.get_embedding(texts)
        print(f"FinBERT Embeddings shape: {embeddings.shape}")
        
    except Exception as e:
        print(f"FinBERT verification failed (likely connection): {e}")
        print("Falling back to Mock...")
        extractor = MockLLMExtractor()
        embeddings = extractor.get_embedding(["Test"])
        print(f"Mock Embeddings shape: {embeddings.shape}")
