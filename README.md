# DataCuration

A comprehensive pipeline to curate data for training, testing, and evaluating large language models.

## Overview

DataCuration provides a modular framework for processing, cleaning, filtering, and transforming datasets to prepare them for machine learning applications. The pipeline supports various data sources, transformation operations, and quality assessment metrics, with a focus on semantic understanding and diversity.

## Features

- **Modular Pipeline Architecture**
  - Pluggable components for data processing
  - Clean separation of concerns with well-defined interfaces
  - Extensible design for custom components

- **Flexible Data Processing**
  - Support for multiple data sources and formats
  - Configurable filtering and transformation rules
  - Quality metrics and validation

- **Semantic Operations**
  - Duplicate detection based on semantic similarity
  - Diversity sampling through clustering
  - Content ranking by similarity to reference examples
  - Embedding-based text analysis

- **Evaluation Framework**
  - Compare different pipeline configurations
  - Measure data quality and diversity
  - Benchmark pipeline performance
  - Identify optimal configurations for specific use cases

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/datacuration.git
cd datacuration

# Install the package and dependencies
pip install -e .
```

## Quick Start

### Basic Text Processing

```python
from datacuration.examples.simple_pipeline import run_example_pipeline

# Process a text file with basic cleaning and filtering
stats = run_example_pipeline(
    input_file="path/to/input.txt",
    output_file="path/to/output.json",
    min_length=5  # Minimum text length to keep
)
```

### Semantic Processing

```python
from datacuration.examples.semantic_pipeline import run_semantic_pipeline

# Process text with semantic duplicate detection and diversity sampling
stats = run_semantic_pipeline(
    input_file="path/to/input.json",
    output_file="path/to/output.json",
    similarity_threshold=0.85,  # Threshold for duplicate detection
    n_clusters=10,              # Number of diversity clusters
    samples_per_cluster=5       # Samples to keep per cluster
)
```

### Pipeline Evaluation

```python
from datacuration.examples.evaluation_example import run_evaluation_example

# Compare different pipeline configurations
run_evaluation_example("path/to/input.json")
```

## Core Components

### Data Sources

Data sources provide input data to the pipeline:

```python
from datacuration.interfaces import DataSource

class MyCustomSource(DataSource):
    def get_data(self):
        # Implement data retrieval logic
        return my_data
        
    def get_metadata(self):
        return {"source_type": "custom", "item_count": len(my_data)}
```

### Data Processors

Processors perform general data processing operations:

```python
from datacuration.interfaces import DataProcessor

class MyProcessor(DataProcessor):
    def process(self, data):
        # Implement processing logic
        return processed_data
        
    def get_stats(self):
        return {"processed_items": len(processed_data)}
```

### Data Filters

Filters remove unwanted data items:

```python
from datacuration.interfaces import DataFilter

class MyFilter(DataFilter):
    def filter(self, data):
        # Implement filtering logic
        return filtered_data
        
    def get_filter_stats(self):
        return {"removed_items": len(data) - len(filtered_data)}
```

### Data Transformers

Transformers modify data items:

```python
from datacuration.interfaces import DataTransformer

class MyTransformer(DataTransformer):
    def transform(self, data):
        # Implement transformation logic
        return transformed_data
```

### Quality Checkers

Quality checkers assess data quality:

```python
from datacuration.interfaces import QualityChecker

class MyQualityChecker(QualityChecker):
    def check_quality(self, data):
        # Implement quality assessment logic
        return {"quality_score": calculate_score(data)}
```

### Data Sinks

Sinks output processed data:

```python
from datacuration.interfaces import DataSink

class MySink(DataSink):
    def write(self, data, metadata=None):
        # Implement data output logic
        save_to_destination(data, metadata)
```

## Building a Pipeline

```python
from datacuration.pipeline import Pipeline
from datacuration.semantic import SimpleEmbeddingProvider, SemanticDuplicateFilter

# Create a pipeline
pipeline = Pipeline(name="my_pipeline")

# Add components
pipeline.set_source(MyCustomSource())
pipeline.add_processor(MyProcessor())
pipeline.add_filter(SemanticDuplicateFilter(SimpleEmbeddingProvider()))
pipeline.add_transformer(MyTransformer())
pipeline.add_quality_checker(MyQualityChecker())
pipeline.add_sink(MySink())

# Run the pipeline
stats = pipeline.run()
```

## Configuration

The pipeline can be configured using YAML files:

```python
from datacuration.config import Config

# Load configuration
config = Config("path/to/config.yaml")

# Access configuration values
source_path = config.get("source.path")
threshold = config.get("filter.threshold", 0.8)  # With default value
```

Example YAML configuration:

```yaml
source:
  type: json
  path: data/input.json
  
filters:
  - type: semantic_duplicate
    threshold: 0.85
    
transformers:
  - type: diversity_sampler
    clusters: 10
    samples_per_cluster: 5
```

## Advanced Usage

See the `examples` directory for more advanced usage patterns and the `ARCHITECTURE.md` file for detailed design information.

## License

This project is licensed under the terms of the LICENSE file included in the repository.
