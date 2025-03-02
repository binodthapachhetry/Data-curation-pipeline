"""Core pipeline implementation for data curation."""

from typing import Any, Dict, List, Optional, Union

from datacuration.interfaces import (
    DataFilter, DataProcessor, DataSink, DataSource, 
    DataTransformer, QualityChecker
)


class Pipeline:
    """Main pipeline for data curation.
    
    The pipeline orchestrates the flow of data through various processing
    components including sources, processors, filters, transformers,
    quality checkers, and sinks.
    """
    
    def __init__(self, name: str = "default"):
        """Initialize a new pipeline.
        
        Args:
            name: A name for the pipeline.
        """
        self.name = name
        self.source: Optional[DataSource] = None
        self.processors: List[DataProcessor] = []
        self.filters: List[DataFilter] = []
        self.transformers: List[DataTransformer] = []
        self.quality_checkers: List[QualityChecker] = []
        self.sinks: List[DataSink] = []
        self.stats: Dict[str, Any] = {}
    
    def set_source(self, source: DataSource) -> "Pipeline":
        """Set the data source for the pipeline.
        
        Args:
            source: The data source.
            
        Returns:
            Pipeline: The pipeline instance for method chaining.
        """
        self.source = source
        return self
    
    def add_processor(self, processor: DataProcessor) -> "Pipeline":
        """Add a data processor to the pipeline.
        
        Args:
            processor: The data processor to add.
            
        Returns:
            Pipeline: The pipeline instance for method chaining.
        """
        self.processors.append(processor)
        return self
    
    def add_filter(self, filter_: DataFilter) -> "Pipeline":
        """Add a data filter to the pipeline.
        
        Args:
            filter_: The data filter to add.
            
        Returns:
            Pipeline: The pipeline instance for method chaining.
        """
        self.filters.append(filter_)
        return self
    
    def add_transformer(self, transformer: DataTransformer) -> "Pipeline":
        """Add a data transformer to the pipeline.
        
        Args:
            transformer: The data transformer to add.
            
        Returns:
            Pipeline: The pipeline instance for method chaining.
        """
        self.transformers.append(transformer)
        return self
    
    def add_quality_checker(self, checker: QualityChecker) -> "Pipeline":
        """Add a quality checker to the pipeline.
        
        Args:
            checker: The quality checker to add.
            
        Returns:
            Pipeline: The pipeline instance for method chaining.
        """
        self.quality_checkers.append(checker)
        return self
    
    def add_sink(self, sink: DataSink) -> "Pipeline":
        """Add a data sink to the pipeline.
        
        Args:
            sink: The data sink to add.
            
        Returns:
            Pipeline: The pipeline instance for method chaining.
        """
        self.sinks.append(sink)
        return self
    
    def run(self) -> Dict[str, Any]:
        """Execute the pipeline.
        
        Returns:
            Dict[str, Any]: Statistics and metadata about the pipeline execution.
        """
        if not self.source:
            raise ValueError("Pipeline requires a data source")
        
        # Get data from source
        data = self.source.get_data()
        metadata = self.source.get_metadata()
        self.stats["source"] = metadata
        
        # Apply processors
        for i, processor in enumerate(self.processors):
            data = processor.process(data)
            self.stats[f"processor_{i}"] = processor.get_stats()
        
        # Apply filters
        for i, filter_ in enumerate(self.filters):
            data = filter_.filter(data)
            self.stats[f"filter_{i}"] = filter_.get_filter_stats()
        
        # Apply transformers
        for i, transformer in enumerate(self.transformers):
            data = transformer.transform(data)
            # Store transformer stats if available
            if hasattr(transformer, 'get_stats'):
                self.stats[f"transformer_{i}"] = transformer.get_stats()
        
        # Check quality
        quality_metrics = {}
        for i, checker in enumerate(self.quality_checkers):
            metrics = checker.check_quality(data)
            quality_metrics[f"checker_{i}"] = metrics
        
        self.stats["quality"] = quality_metrics
        
        # Store the final data for evaluation purposes
        self.final_data = data
        
        # Write to sinks
        for sink in self.sinks:
            sink.write(data, metadata=self.stats)
        
        return self.stats
