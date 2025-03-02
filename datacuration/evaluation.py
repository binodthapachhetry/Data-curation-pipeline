"""Evaluation framework for data curation pipelines."""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import time
import numpy as np
from collections import defaultdict

from datacuration.interfaces import DataSource
from datacuration.pipeline import Pipeline


class PipelineEvaluator:
    """Evaluator for comparing and benchmarking pipeline configurations."""
    
    def __init__(self):
        """Initialize the pipeline evaluator."""
        self.results = {}
        self.metrics = {}
    
    def evaluate_pipeline(
        self, 
        pipeline: Pipeline, 
        metrics: List[Callable[[Any, Dict[str, Any]], float]] = None,
        name: str = None
    ) -> Dict[str, Any]:
        """Evaluate a pipeline using specified metrics.
        
        Args:
            pipeline: The pipeline to evaluate.
            metrics: List of metric functions that take (data, stats) and return a score.
            name: Optional name for this evaluation run.
            
        Returns:
            Dict[str, Any]: Evaluation results including metrics and timing.
        """
        if name is None:
            name = pipeline.name
            
        if metrics is None:
            metrics = []
            
        # Time the pipeline execution
        start_time = time.time()
        stats = pipeline.run()
        end_time = time.time()
        
        # Calculate execution time
        execution_time = end_time - start_time
        
        # Get the final data from the pipeline
        # This requires modifying the pipeline to store its final output
        final_data = getattr(pipeline, 'final_data', None)
        
        # Calculate metrics
        metric_results = {}
        for i, metric_fn in enumerate(metrics):
            try:
                if final_data is not None:
                    score = metric_fn(final_data, stats)
                    metric_name = getattr(metric_fn, '__name__', f'metric_{i}')
                    metric_results[metric_name] = score
            except Exception as e:
                metric_results[f'metric_{i}_error'] = str(e)
        
        # Compile results
        results = {
            'name': name,
            'execution_time': execution_time,
            'stats': stats,
            'metrics': metric_results
        }
        
        # Store results
        self.results[name] = results
        
        return results
    
    def compare_pipelines(
        self, 
        pipelines: List[Pipeline],
        metrics: List[Callable[[Any, Dict[str, Any]], float]] = None,
        names: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple pipeline configurations.
        
        Args:
            pipelines: List of pipelines to compare.
            metrics: List of metric functions to apply to each pipeline.
            names: Optional list of names for the pipelines.
            
        Returns:
            Dict[str, Dict[str, Any]]: Comparison results for all pipelines.
        """
        if names is None:
            names = [p.name for p in pipelines]
            
        results = {}
        for i, pipeline in enumerate(pipelines):
            name = names[i]
            results[name] = self.evaluate_pipeline(pipeline, metrics, name)
            
        return results
    
    def get_best_pipeline(self, metric_name: str, higher_is_better: bool = True) -> Tuple[str, Dict[str, Any]]:
        """Get the best pipeline based on a specific metric.
        
        Args:
            metric_name: Name of the metric to use for comparison.
            higher_is_better: Whether higher metric values are better.
            
        Returns:
            Tuple[str, Dict[str, Any]]: Name and results of the best pipeline.
        """
        if not self.results:
            raise ValueError("No pipeline evaluations available")
            
        best_name = None
        best_value = float('-inf') if higher_is_better else float('inf')
        best_results = None
        
        for name, results in self.results.items():
            metric_value = results.get('metrics', {}).get(metric_name)
            
            if metric_value is None:
                continue
                
            if higher_is_better and metric_value > best_value:
                best_value = metric_value
                best_name = name
                best_results = results
            elif not higher_is_better and metric_value < best_value:
                best_value = metric_value
                best_name = name
                best_results = results
                
        if best_name is None:
            raise ValueError(f"Metric '{metric_name}' not found in any evaluation results")
            
        return best_name, best_results
    
    def summarize_results(self) -> Dict[str, Any]:
        """Summarize all evaluation results.
        
        Returns:
            Dict[str, Any]: Summary of all evaluation results.
        """
        if not self.results:
            return {"status": "No evaluations performed"}
            
        summary = {
            "pipeline_count": len(self.results),
            "pipelines": list(self.results.keys()),
            "execution_times": {},
            "metrics": defaultdict(dict)
        }
        
        # Collect execution times
        for name, results in self.results.items():
            summary["execution_times"][name] = results.get("execution_time")
            
            # Collect metrics
            for metric_name, value in results.get("metrics", {}).items():
                summary["metrics"][metric_name][name] = value
                
        # Calculate averages, mins, maxes for each metric
        for metric_name, values in summary["metrics"].items():
            if values:
                summary["metrics"][metric_name]["avg"] = np.mean(list(values.values()))
                summary["metrics"][metric_name]["min"] = min(values.values())
                summary["metrics"][metric_name]["max"] = max(values.values())
                
        return summary


# Common evaluation metrics

def data_reduction_ratio(data: Any, stats: Dict[str, Any]) -> float:
    """Calculate the ratio of output data size to input data size.
    
    Args:
        data: The output data.
        stats: Pipeline statistics.
        
    Returns:
        float: Ratio of output size to input size (lower is more reduction).
    """
    input_count = stats.get('source', {}).get('item_count', 0)
    if input_count == 0:
        return 1.0
        
    output_count = len(data) if isinstance(data, (list, tuple)) else 1
    return output_count / input_count


def semantic_diversity_score(data: List[str], stats: Dict[str, Any]) -> float:
    """Estimate semantic diversity of the dataset.
    
    This is a placeholder for a more sophisticated implementation that would
    actually compute semantic diversity using embeddings.
    
    Args:
        data: The output data.
        stats: Pipeline statistics.
        
    Returns:
        float: Estimated semantic diversity score (higher is more diverse).
    """
    # In a real implementation, this would compute embeddings and measure
    # the average distance between samples or other diversity metrics
    
    # For now, we'll use a simple proxy based on cluster stats if available
    cluster_stats = stats.get('transformer_0', {}).get('cluster_sizes', {})
    if cluster_stats:
        # If we have cluster information, use the evenness of cluster sizes as a proxy
        cluster_sizes = list(cluster_stats.values())
        if not cluster_sizes:
            return 0.0
            
        # Calculate entropy of cluster size distribution as a diversity measure
        total = sum(cluster_sizes)
        if total == 0:
            return 0.0
            
        probabilities = [size / total for size in cluster_sizes]
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probabilities)
        max_entropy = np.log(len(cluster_sizes))
        
        # Normalize to 0-1 range
        if max_entropy == 0:
            return 0.0
        return entropy / max_entropy
    
    # Fallback to a simple proxy based on unique ratio
    if not data:
        return 0.0
        
    unique_ratio = len(set(data)) / len(data)
    return unique_ratio


def quality_score(data: List[str], stats: Dict[str, Any]) -> float:
    """Calculate an aggregate quality score based on available quality metrics.
    
    Args:
        data: The output data.
        stats: Pipeline statistics.
        
    Returns:
        float: Quality score (higher is better).
    """
    quality_metrics = stats.get('quality', {})
    if not quality_metrics:
        return 0.0
        
    # Extract available metrics
    scores = []
    
    # If we have checker metrics, use them
    for checker_key, metrics in quality_metrics.items():
        if isinstance(metrics, dict):
            # Add unique ratio if available
            if 'unique_ratio' in metrics:
                scores.append(metrics['unique_ratio'])
                
    # If no scores were found, return default
    if not scores:
        return 0.5
        
    # Return average of available scores
    return sum(scores) / len(scores)
