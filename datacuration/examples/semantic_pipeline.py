"""Example pipeline using semantic components for data curation."""

import json
import os
from typing import Any, Dict, List

from datacuration.interfaces import DataSink, DataSource
from datacuration.pipeline import Pipeline
from datacuration.semantic import (
    SimpleEmbeddingProvider, 
    SemanticDuplicateFilter,
    DiversityClusterSampler,
    SemanticSimilarityRanker
)


class JsonFileSource(DataSource):
    """A data source that reads text from a JSON file."""
    
    def __init__(self, file_path: str, text_field: str = "text"):
        """Initialize the JSON file source.
        
        Args:
            file_path: Path to the JSON file.
            text_field: Field name containing the text in each JSON object.
        """
        self.file_path = file_path
        self.text_field = text_field
        self.data = None
    
    def get_data(self) -> List[str]:
        """Read text from the JSON file.
        
        Returns:
            List[str]: Text items from the JSON file.
        """
        if self.data is None:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
            # Handle both list of objects and list of strings
            if isinstance(json_data, list):
                if json_data and isinstance(json_data[0], dict):
                    self.data = [item.get(self.text_field, "") for item in json_data if self.text_field in item]
                else:
                    self.data = [str(item) for item in json_data]
            else:
                raise ValueError(f"Expected JSON array in {self.file_path}")
                
        return self.data
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the JSON file.
        
        Returns:
            Dict[str, Any]: Metadata about the JSON file.
        """
        data = self.get_data()  # Ensure data is loaded
        return {
            'file_path': self.file_path,
            'file_size': os.path.getsize(self.file_path),
            'item_count': len(data),
        }


def run_semantic_pipeline(
    input_file: str, 
    output_file: str, 
    similarity_threshold: float = 0.9,
    n_clusters: int = 10,
    samples_per_cluster: int = 5,
    reference_text: str = None,
    top_k: int = None
) -> Dict[str, Any]:
    """Run a semantic processing pipeline for text data.
    
    Args:
        input_file: Path to the input JSON file.
        output_file: Path to the output JSON file.
        similarity_threshold: Threshold for duplicate detection (0.0-1.0).
        n_clusters: Number of diversity clusters to create.
        samples_per_cluster: Maximum samples to take from each cluster.
        reference_text: Optional reference text for similarity ranking.
        top_k: Optional limit on number of results when using reference_text.
        
    Returns:
        Dict[str, Any]: Statistics about the pipeline execution.
    """
    pipeline = Pipeline(name="semantic_processing")
    
    # Create embedding provider (shared by all semantic components)
    embedding_provider = SimpleEmbeddingProvider()
    
    # Configure the pipeline
    pipeline.set_source(JsonFileSource(input_file))
    
    # Add semantic duplicate filter
    pipeline.add_filter(
        SemanticDuplicateFilter(
            embedding_provider=embedding_provider,
            similarity_threshold=similarity_threshold
        )
    )
    
    # Add diversity sampler
    pipeline.add_transformer(
        DiversityClusterSampler(
            embedding_provider=embedding_provider,
            n_clusters=n_clusters,
            samples_per_cluster=samples_per_cluster
        )
    )
    
    # Optionally add similarity ranker if reference text is provided
    if reference_text:
        pipeline.add_transformer(
            SemanticSimilarityRanker(
                embedding_provider=embedding_provider,
                reference_text=reference_text,
                top_k=top_k
            )
        )
    
    # Add sink
    pipeline.add_sink(JsonFileSink(output_file))
    
    # Run the pipeline
    stats = pipeline.run()
    
    print(f"Pipeline completed. Processed {stats['source']['item_count']} items.")
    print(f"Removed {stats['filter_0']['duplicate_count']} semantic duplicates.")
    print(f"Output written to {output_file}")
    
    return stats


class JsonFileSink(DataSink):
    """A data sink that writes data to a JSON file."""
    
    def __init__(self, file_path: str):
        """Initialize the JSON file sink.
        
        Args:
            file_path: Path to the output JSON file.
        """
        self.file_path = file_path
    
    def write(self, data: List[str], metadata: Dict[str, Any] = None) -> None:
        """Write data and metadata to a JSON file.
        
        Args:
            data: The data to write.
            metadata: Optional metadata to include.
        """
        output = {
            'data': data,
            'metadata': metadata or {},
        }
        
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run semantic data curation pipeline")
    parser.add_argument("input_file", help="Path to input JSON file")
    parser.add_argument("output_file", help="Path to output JSON file")
    parser.add_argument("--similarity", type=float, default=0.9, 
                        help="Similarity threshold for duplicate detection (0.0-1.0)")
    parser.add_argument("--clusters", type=int, default=10,
                        help="Number of diversity clusters")
    parser.add_argument("--samples", type=int, default=5,
                        help="Maximum samples per cluster")
    parser.add_argument("--reference", type=str, default=None,
                        help="Reference text for similarity ranking")
    parser.add_argument("--top", type=int, default=None,
                        help="Top K results when using reference text")
    
    args = parser.parse_args()
    
    run_semantic_pipeline(
        args.input_file,
        args.output_file,
        similarity_threshold=args.similarity,
        n_clusters=args.clusters,
        samples_per_cluster=args.samples,
        reference_text=args.reference,
        top_k=args.top
    )
