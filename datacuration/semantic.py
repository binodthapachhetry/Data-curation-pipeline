"""Semantic components for data curation using embeddings."""

from abc import ABC
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from datacuration.interfaces import DataFilter, DataTransformer


class EmbeddingProvider(ABC):
    """Base class for embedding providers that convert text to vector representations."""
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Convert a list of texts to embeddings.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            np.ndarray: Matrix of embeddings with shape (len(texts), embedding_dim).
        """
        raise NotImplementedError("Embedding providers must implement get_embeddings")


class SimpleEmbeddingProvider(EmbeddingProvider):
    """A simple embedding provider that uses a pre-trained model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding provider with a model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers package is required. "
                "Install it with: pip install sentence-transformers"
            )
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Convert a list of texts to embeddings using sentence-transformers.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            np.ndarray: Matrix of embeddings with shape (len(texts), embedding_dim).
        """
        return self.model.encode(texts, convert_to_numpy=True)


class SemanticDuplicateFilter(DataFilter):
    """Filter that removes semantic duplicates based on embedding similarity."""
    
    def __init__(
        self, 
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float = 0.9,
        batch_size: int = 1000
    ):
        """Initialize the semantic duplicate filter.
        
        Args:
            embedding_provider: Provider for text embeddings.
            similarity_threshold: Threshold above which items are considered duplicates (0.0-1.0).
            batch_size: Size of batches for processing large datasets.
        """
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.stats = {
            'input_count': 0,
            'output_count': 0,
            'duplicate_count': 0,
            'similarity_threshold': similarity_threshold,
        }
    
    def filter(self, data: List[str]) -> List[str]:
        """Filter out semantic duplicates from the data.
        
        Args:
            data: List of text items.
            
        Returns:
            List[str]: Filtered list with duplicates removed.
        """
        self.stats['input_count'] = len(data)
        
        if len(data) == 0:
            self.stats['output_count'] = 0
            return []
        
        # For very small datasets, process all at once
        if len(data) <= self.batch_size:
            filtered_data = self._process_batch(data)
        else:
            # Process in batches for larger datasets
            filtered_data = []
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size]
                filtered_batch = self._process_batch(batch)
                filtered_data.extend(filtered_batch)
        
        self.stats['output_count'] = len(filtered_data)
        self.stats['duplicate_count'] = self.stats['input_count'] - self.stats['output_count']
        
        return filtered_data
    
    def _process_batch(self, batch: List[str]) -> List[str]:
        """Process a batch of data to remove semantic duplicates.
        
        Args:
            batch: A batch of text items.
            
        Returns:
            List[str]: Filtered batch with duplicates removed.
        """
        if not batch:
            return []
            
        # Get embeddings for the batch
        embeddings = self.embedding_provider.get_embeddings(batch)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Set diagonal to 0 to avoid self-matches
        np.fill_diagonal(similarities, 0)
        
        # Find duplicates
        keep_indices = []
        for i in range(len(batch)):
            # If this item hasn't been marked as a duplicate of a previous item
            if i not in keep_indices:
                # Add it to our keep list
                keep_indices.append(i)
                
                # Find all items that are too similar to this one
                duplicates = np.where(similarities[i] > self.similarity_threshold)[0]
                
                # Don't mark any items we've already decided to keep
                duplicates = [d for d in duplicates if d not in keep_indices]
                
                # We don't need to do anything with duplicates here, just not add them to keep_indices
        
        # Return only the items we're keeping
        return [batch[i] for i in sorted(keep_indices)]
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get statistics about the filtering operation.
        
        Returns:
            Dict[str, Any]: Statistics about the filtering operation.
        """
        return self.stats


class DiversityClusterSampler(DataTransformer):
    """Transformer that samples diverse examples using clustering."""
    
    def __init__(
        self, 
        embedding_provider: EmbeddingProvider,
        n_clusters: int = 10,
        samples_per_cluster: Optional[int] = None,
        random_state: int = 42
    ):
        """Initialize the diversity cluster sampler.
        
        Args:
            embedding_provider: Provider for text embeddings.
            n_clusters: Number of clusters to create.
            samples_per_cluster: Maximum samples to take from each cluster.
                If None, all samples in each cluster are kept.
            random_state: Random seed for reproducibility.
        """
        self.embedding_provider = embedding_provider
        self.n_clusters = n_clusters
        self.samples_per_cluster = samples_per_cluster
        self.random_state = random_state
        self.cluster_stats = {}
    
    def transform(self, data: List[str]) -> List[str]:
        """Transform the data by sampling diverse examples.
        
        Args:
            data: List of text items.
            
        Returns:
            List[str]: Sampled diverse examples.
        """
        if len(data) == 0:
            return []
            
        # Adjust n_clusters if we have fewer data points than requested clusters
        actual_n_clusters = min(self.n_clusters, len(data))
        
        # Get embeddings
        embeddings = self.embedding_provider.get_embeddings(data)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=actual_n_clusters, 
            random_state=self.random_state,
            n_init=10
        )
        clusters = kmeans.fit_predict(embeddings)
        
        # Sample from each cluster
        sampled_indices = []
        self.cluster_stats = {'cluster_sizes': {}}
        
        for cluster_id in range(actual_n_clusters):
            # Get indices of all items in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            self.cluster_stats['cluster_sizes'][cluster_id] = len(cluster_indices)
            
            # If we need to sample, do so randomly
            if self.samples_per_cluster is not None and len(cluster_indices) > self.samples_per_cluster:
                np.random.seed(self.random_state + cluster_id)  # Ensure reproducibility
                sampled_from_cluster = np.random.choice(
                    cluster_indices, 
                    size=self.samples_per_cluster, 
                    replace=False
                )
                sampled_indices.extend(sampled_from_cluster)
            else:
                # Otherwise take all examples from this cluster
                sampled_indices.extend(cluster_indices)
        
        # Return the sampled items
        sampled_data = [data[i] for i in sorted(sampled_indices)]
        self.cluster_stats['input_count'] = len(data)
        self.cluster_stats['output_count'] = len(sampled_data)
        
        return sampled_data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the sampling operation.
        
        Returns:
            Dict[str, Any]: Statistics about the sampling operation.
        """
        return self.cluster_stats


class SemanticSimilarityRanker(DataTransformer):
    """Transformer that ranks items by similarity to a reference text."""
    
    def __init__(
        self, 
        embedding_provider: EmbeddingProvider,
        reference_text: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ):
        """Initialize the semantic similarity ranker.
        
        Args:
            embedding_provider: Provider for text embeddings.
            reference_text: The reference text to compare against.
            top_k: If provided, only return the top k most similar items.
            similarity_threshold: If provided, only return items with
                similarity above this threshold.
        """
        self.embedding_provider = embedding_provider
        self.reference_text = reference_text
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.similarity_scores = []
    
    def transform(self, data: List[str]) -> List[str]:
        """Transform the data by ranking items by similarity to the reference.
        
        Args:
            data: List of text items.
            
        Returns:
            List[str]: Ranked and filtered items.
        """
        if len(data) == 0:
            return []
        
        # Get reference embedding
        reference_embedding = self.embedding_provider.get_embeddings([self.reference_text])[0]
        
        # Get embeddings for all items
        embeddings = self.embedding_provider.get_embeddings(data)
        
        # Calculate similarities
        similarities = cosine_similarity([reference_embedding], embeddings)[0]
        
        # Create (index, similarity) pairs
        indexed_similarities = list(enumerate(similarities))
        
        # Sort by similarity (descending)
        indexed_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Store similarity scores for stats
        self.similarity_scores = [(data[idx], score) for idx, score in indexed_similarities]
        
        # Apply threshold if specified
        if self.similarity_threshold is not None:
            indexed_similarities = [
                (idx, score) for idx, score in indexed_similarities 
                if score >= self.similarity_threshold
            ]
        
        # Apply top_k if specified
        if self.top_k is not None:
            indexed_similarities = indexed_similarities[:self.top_k]
        
        # Extract indices in order of similarity
        indices = [idx for idx, _ in indexed_similarities]
        
        # Return items in order of similarity
        return [data[idx] for idx in indices]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the ranking operation.
        
        Returns:
            Dict[str, Any]: Statistics about the ranking operation.
        """
        return {
            'reference_text': self.reference_text,
            'input_count': len(self.similarity_scores),
            'output_count': len(self.similarity_scores),
            'max_similarity': max([score for _, score in self.similarity_scores]) if self.similarity_scores else 0,
            'min_similarity': min([score for _, score in self.similarity_scores]) if self.similarity_scores else 0,
            'avg_similarity': sum([score for _, score in self.similarity_scores]) / len(self.similarity_scores) if self.similarity_scores else 0,
        }
