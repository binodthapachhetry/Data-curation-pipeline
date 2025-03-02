"""Example demonstrating the evaluation framework for data curation pipelines."""

import json
import os
from typing import List, Dict, Any

from datacuration.interfaces import DataSource
from datacuration.pipeline import Pipeline
from datacuration.semantic import (
    SimpleEmbeddingProvider, 
    SemanticDuplicateFilter,
    DiversityClusterSampler
)
from datacuration.evaluation import (
    PipelineEvaluator,
    data_reduction_ratio,
    semantic_diversity_score,
    quality_score
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


class MemorySink:
    """A sink that just stores data in memory for evaluation purposes."""
    
    def __init__(self):
        """Initialize the memory sink."""
        self.data = None
        self.metadata = None
    
    def write(self, data, metadata=None):
        """Store the data and metadata in memory.
        
        Args:
            data: The data to store.
            metadata: Optional metadata to store.
        """
        self.data = data
        self.metadata = metadata


def create_pipeline(
    input_file: str,
    name: str,
    similarity_threshold: float,
    n_clusters: int,
    samples_per_cluster: int
) -> Pipeline:
    """Create a pipeline with the specified configuration.
    
    Args:
        input_file: Path to the input JSON file.
        name: Name for the pipeline.
        similarity_threshold: Threshold for duplicate detection.
        n_clusters: Number of diversity clusters.
        samples_per_cluster: Maximum samples per cluster.
        
    Returns:
        Pipeline: Configured pipeline.
    """
    pipeline = Pipeline(name=name)
    
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
    
    # Add memory sink for evaluation
    pipeline.add_sink(MemorySink())
    
    return pipeline


def run_evaluation_example(input_file: str):
    """Run an example evaluation of multiple pipeline configurations.
    
    Args:
        input_file: Path to the input JSON file.
    """
    # Create evaluator
    evaluator = PipelineEvaluator()
    
    # Create pipelines with different configurations
    pipelines = [
        create_pipeline(
            input_file=input_file,
            name="high_precision",
            similarity_threshold=0.9,  # High threshold = fewer duplicates removed
            n_clusters=10,
            samples_per_cluster=5
        ),
        create_pipeline(
            input_file=input_file,
            name="balanced",
            similarity_threshold=0.8,  # Medium threshold
            n_clusters=8,
            samples_per_cluster=10
        ),
        create_pipeline(
            input_file=input_file,
            name="high_diversity",
            similarity_threshold=0.7,  # Low threshold = more duplicates removed
            n_clusters=15,
            samples_per_cluster=3
        )
    ]
    
    # Define metrics for evaluation
    metrics = [
        data_reduction_ratio,
        semantic_diversity_score,
        quality_score
    ]
    
    # Compare pipelines
    results = evaluator.compare_pipelines(
        pipelines=pipelines,
        metrics=metrics
    )
    
    # Print results
    print("\n=== Pipeline Evaluation Results ===\n")
    for name, result in results.items():
        print(f"Pipeline: {name}")
        print(f"  Execution time: {result['execution_time']:.2f} seconds")
        print("  Metrics:")
        for metric_name, value in result['metrics'].items():
            print(f"    {metric_name}: {value:.4f}")
        print()
    
    # Get summary
    summary = evaluator.summarize_results()
    print("\n=== Evaluation Summary ===\n")
    print(f"Number of pipelines evaluated: {summary['pipeline_count']}")
    
    print("\nMetric Comparison:")
    for metric_name, values in summary['metrics'].items():
        print(f"  {metric_name}:")
        print(f"    Average: {values['avg']:.4f}")
        print(f"    Min: {values['min']:.4f} ({min(values.items(), key=lambda x: x[1] if x[0] not in ['avg', 'min', 'max'] else float('inf'))[0]})")
        print(f"    Max: {values['max']:.4f} ({max(values.items(), key=lambda x: x[1] if x[0] not in ['avg', 'min', 'max'] else float('-inf'))[0]})")
    
    # Find best pipeline for each metric
    print("\nBest Pipelines:")
    
    # For data_reduction_ratio, lower is better
    try:
        best_reduction, _ = evaluator.get_best_pipeline("data_reduction_ratio", higher_is_better=False)
        print(f"  Best for data reduction: {best_reduction}")
    except ValueError:
        pass
    
    # For diversity and quality, higher is better
    try:
        best_diversity, _ = evaluator.get_best_pipeline("semantic_diversity_score", higher_is_better=True)
        print(f"  Best for semantic diversity: {best_diversity}")
    except ValueError:
        pass
    
    try:
        best_quality, _ = evaluator.get_best_pipeline("quality_score", higher_is_better=True)
        print(f"  Best for quality: {best_quality}")
    except ValueError:
        pass


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluation_example.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    run_evaluation_example(input_file)
