"""A simple example pipeline for text data curation."""

import json
import os
from typing import Any, Dict, List

from datacuration.interfaces import (
    DataFilter, DataProcessor, DataSink, DataSource, 
    DataTransformer, QualityChecker
)
from datacuration.pipeline import Pipeline


class TextFileSource(DataSource):
    """A data source that reads text from a file."""
    
    def __init__(self, file_path: str):
        """Initialize the text file source.
        
        Args:
            file_path: Path to the text file.
        """
        self.file_path = file_path
    
    def get_data(self) -> List[str]:
        """Read lines from the text file.
        
        Returns:
            List[str]: Lines from the text file.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the text file.
        
        Returns:
            Dict[str, Any]: Metadata about the text file.
        """
        return {
            'file_path': self.file_path,
            'file_size': os.path.getsize(self.file_path),
            'line_count': len(self.get_data()),
        }


class TextCleaningProcessor(DataProcessor):
    """A processor that cleans text data."""
    
    def __init__(self):
        """Initialize the text cleaning processor."""
        self.stats = {
            'processed_lines': 0,
            'empty_lines_removed': 0,
        }
    
    def process(self, data: List[str]) -> List[str]:
        """Clean the text data.
        
        Args:
            data: List of text lines.
            
        Returns:
            List[str]: Cleaned text lines.
        """
        cleaned_lines = []
        for line in data:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
            else:
                self.stats['empty_lines_removed'] += 1
        
        self.stats['processed_lines'] = len(data)
        return cleaned_lines
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cleaning operation.
        
        Returns:
            Dict[str, Any]: Statistics about the cleaning operation.
        """
        return self.stats


class MinLengthFilter(DataFilter):
    """A filter that removes text lines shorter than a minimum length."""
    
    def __init__(self, min_length: int):
        """Initialize the minimum length filter.
        
        Args:
            min_length: The minimum length of text lines to keep.
        """
        self.min_length = min_length
        self.stats = {
            'input_lines': 0,
            'output_lines': 0,
            'filtered_lines': 0,
        }
    
    def filter(self, data: List[str]) -> List[str]:
        """Filter text lines by minimum length.
        
        Args:
            data: List of text lines.
            
        Returns:
            List[str]: Filtered text lines.
        """
        self.stats['input_lines'] = len(data)
        filtered_data = [line for line in data if len(line) >= self.min_length]
        self.stats['output_lines'] = len(filtered_data)
        self.stats['filtered_lines'] = self.stats['input_lines'] - self.stats['output_lines']
        return filtered_data
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get statistics about the filtering operation.
        
        Returns:
            Dict[str, Any]: Statistics about the filtering operation.
        """
        return self.stats


class TextLowercaseTransformer(DataTransformer):
    """A transformer that converts text to lowercase."""
    
    def transform(self, data: List[str]) -> List[str]:
        """Convert text to lowercase.
        
        Args:
            data: List of text lines.
            
        Returns:
            List[str]: Lowercase text lines.
        """
        return [line.lower() for line in data]


class TextQualityChecker(QualityChecker):
    """A quality checker for text data."""
    
    def check_quality(self, data: List[str]) -> Dict[str, float]:
        """Check the quality of text data.
        
        Args:
            data: List of text lines.
            
        Returns:
            Dict[str, float]: Quality metrics for the text data.
        """
        if not data:
            return {'avg_length': 0.0, 'unique_ratio': 1.0}
        
        total_length = sum(len(line) for line in data)
        avg_length = total_length / len(data)
        unique_lines = len(set(data))
        unique_ratio = unique_lines / len(data)
        
        return {
            'avg_length': avg_length,
            'unique_ratio': unique_ratio,
        }


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


def run_example_pipeline(input_file: str, output_file: str, min_length: int = 5) -> Dict[str, Any]:
    """Run an example text processing pipeline.
    
    Args:
        input_file: Path to the input text file.
        output_file: Path to the output JSON file.
        min_length: Minimum length of text lines to keep.
        
    Returns:
        Dict[str, Any]: Statistics about the pipeline execution.
    """
    pipeline = Pipeline(name="text_processing")
    
    # Configure the pipeline
    pipeline.set_source(TextFileSource(input_file))
    pipeline.add_processor(TextCleaningProcessor())
    pipeline.add_filter(MinLengthFilter(min_length))
    pipeline.add_transformer(TextLowercaseTransformer())
    pipeline.add_quality_checker(TextQualityChecker())
    pipeline.add_sink(JsonFileSink(output_file))
    
    # Run the pipeline
    stats = pipeline.run()
    
    print(f"Pipeline completed. Processed {stats['source']['line_count']} lines.")
    print(f"Output written to {output_file}")
    
    return stats


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python simple_pipeline.py <input_file> <output_file> [min_length]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    min_length = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    run_example_pipeline(input_file, output_file, min_length)
