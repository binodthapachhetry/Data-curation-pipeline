"""Core interfaces for the DataCuration pipeline components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class DataSource(ABC):
    """Interface for data sources that provide data to the pipeline."""
    
    @abstractmethod
    def get_data(self) -> Any:
        """Retrieve data from the source.
        
        Returns:
            Any: The data in a format that can be processed by the pipeline.
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data source.
        
        Returns:
            Dict[str, Any]: Metadata describing the data source.
        """
        pass


class DataProcessor(ABC):
    """Interface for components that process data in the pipeline."""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process the input data.
        
        Args:
            data: The input data to process.
            
        Returns:
            Any: The processed data.
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing operation.
        
        Returns:
            Dict[str, Any]: Statistics about the processing operation.
        """
        pass


class DataFilter(ABC):
    """Interface for components that filter data in the pipeline."""
    
    @abstractmethod
    def filter(self, data: Any) -> Any:
        """Filter the input data.
        
        Args:
            data: The input data to filter.
            
        Returns:
            Any: The filtered data.
        """
        pass
    
    @abstractmethod
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get statistics about the filtering operation.
        
        Returns:
            Dict[str, Any]: Statistics about the filtering operation.
        """
        pass


class DataTransformer(ABC):
    """Interface for components that transform data in the pipeline."""
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Transform the input data.
        
        Args:
            data: The input data to transform.
            
        Returns:
            Any: The transformed data.
        """
        pass


class DataSink(ABC):
    """Interface for components that store or export processed data."""
    
    @abstractmethod
    def write(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Write data to the sink.
        
        Args:
            data: The data to write.
            metadata: Optional metadata to associate with the data.
        """
        pass


class QualityChecker(ABC):
    """Interface for components that assess data quality."""
    
    @abstractmethod
    def check_quality(self, data: Any) -> Dict[str, float]:
        """Check the quality of the data.
        
        Args:
            data: The data to check.
            
        Returns:
            Dict[str, float]: Quality metrics for the data.
        """
        pass
