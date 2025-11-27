from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """
    Wrapper around a SentenceTransformer model to generate dense embeddings.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode a single string or list of strings into a numpy array of embeddings.
        Shape:
          - single string -> (1, dim)
          - list of strings -> (N, dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings
