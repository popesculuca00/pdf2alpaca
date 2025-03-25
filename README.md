# 📚 Instruction Dataset Generator 🤖

A tool for converting PDF documents into instruction-response pairs suitable for fine-tuning language models in the Alpaca format.

## 🌟 Features

- 📄 **PDF Processing**: Extract text from PDF documents with smart chunking and boundary detection
- 🧠 **AI-Powered Generation**: Use Gemini 1.5 Pro to create high-quality instruction-response pairs
- 🔍 **Data Validation**: Ensure all generated pairs follow the Alpaca format
- 💾 **Dataset Formatting**: Save properly formatted datasets ready for model fine-tuning
- ⚡ **Parallel Processing**: Process multiple PDFs simultaneously for faster throughput
- 🔄 **Robust Error Handling**: Retry logic, rate limiting, and comprehensive error reporting
- 🛠️ **Advanced Utilities**: Merge datasets, convert to JSONL format, and create backups

## 🧠 Theory and Approach

### Alpaca Format for LLM Fine-tuning

The Alpaca format is a standardized structure for creating instruction-following datasets. It originated from the Stanford Alpaca project, which aimed to create an open-source variant of LLaMA fine-tuned to follow instructions similar to ChatGPT.

Key aspects of the Alpaca format:
- **Instruction-Response Pattern**: Structures data in a way that teaches models to follow explicit instructions
- **Three-Part Structure**: 
  - `instruction`: The task or question posed to the model
  - `input`: Additional context or information (can be empty)
  - `output`: The expected response from the model
- **Fine-tuning Benefits**: Models fine-tuned on Alpaca-formatted data tend to:
  - Follow user instructions more precisely
  - Generate more helpful and contextually appropriate responses
  - Better understand the intent behind user queries

This format has become a standard for instruction-tuning across various LLM frameworks, enabling models to better understand and execute user commands.

For a deeper dive into Alpaca formats, see the [Stanford Alpaca project](https://github.com/tatsu-lab/stanford_alpaca) and [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) which provides code for reproducing Stanford Alpaca results using low-rank adaptation.

### Text Chunking Strategy

Processing lengthy documents presents challenges for LLMs due to context window limitations. Our chunking approach breaks down documents into manageable pieces while preserving context:

- **Why Chunk?**
  - Large documents exceed LLM context windows
  - Processing smaller chunks allows for more focused and relevant instruction-response pairs
  - Reduces complexity and improves generation quality

- **Intelligent Boundary Detection**:
  - Prioritizes breaking at paragraph boundaries (most natural)
  - Falls back to sentence boundaries when paragraphs are too long
  - Ensures semantic coherence within each chunk

### The Importance of Overlap

Overlap between chunks is crucial for maintaining contextual continuity:

- **Preserving Context**: Information often spans natural text boundaries
- **Avoiding Information Loss**: Without overlap, content at chunk boundaries might be misinterpreted or lost
- **Cross-References**: Many documents contain internal references that need context from previous sections
- **Optimal Overlap**: 10-20% overlap typically balances context preservation with computational efficiency

#### Chunking & Overlap Example

Consider this text from a research paper:

```
Machine learning models have revolutionized many fields. Deep neural networks 
in particular have shown remarkable results in computer vision tasks. 
Convolutional Neural Networks (CNNs) are specifically designed for processing 
grid-like data such as images.

CNNs use convolutional layers to extract features from input data. These layers 
apply filters across the input, detecting patterns regardless of their position. 
This property, known as translation invariance, is crucial for image recognition.

After feature extraction, the model typically includes pooling layers to reduce 
dimensionality. Finally, fully connected layers produce the output predictions.
```

Without overlap, if we chunk with a size of 40 words:

**Chunk 1:**
```
Machine learning models have revolutionized many fields. Deep neural networks 
in particular have shown remarkable results in computer vision tasks. 
Convolutional Neural Networks (CNNs) are specifically designed for processing 
grid-like data such as images.
```

**Chunk 2:**
```
CNNs use convolutional layers to extract features from input data. These layers 
apply filters across the input, detecting patterns regardless of their position. 
This property, known as translation invariance, is crucial for image recognition.
```

**Chunk 3:**
```
After feature extraction, the model typically includes pooling layers to reduce 
dimensionality. Finally, fully connected layers produce the output predictions.
```

With an overlap of ~10 words (25% overlap):

**Chunk 1:**
```
Machine learning models have revolutionized many fields. Deep neural networks 
in particular have shown remarkable results in computer vision tasks. 
Convolutional Neural Networks (CNNs) are specifically designed for processing 
grid-like data such as images.
```

**Chunk 2:**
```
designed for processing grid-like data such as images. CNNs use convolutional 
layers to extract features from input data. These layers apply filters across 
the input, detecting patterns regardless of their position. This property, 
known as translation invariance, is crucial for image recognition.
```

**Chunk 3:**
```
patterns regardless of their position. This property, known as translation 
invariance, is crucial for image recognition. After feature extraction, the model 
typically includes pooling layers to reduce dimensionality. Finally, fully 
connected layers produce the output predictions.
```

The overlap ensures that when generating instruction-response pairs:
1. The context flows naturally between chunks
2. We don't lose information at the boundaries
3. The model can generate more coherent question-answer pairs about concepts that span chunk boundaries

This approach ensures that generated instruction-response pairs maintain accuracy even when the source information spans chunk boundaries.

## ⚙️ Parameter Selection Guide

The tool's performance, output quality, and resource requirements are significantly influenced by the parameters you choose. Here's a guide to help you select optimal parameters for your specific needs:

### Key Parameters and Their Impact

#### 1. Chunk Size (`--chunk_size`)
- **Description**: The size of text chunks in characters extracted from PDFs
- **Performance Impact**:
  - **Smaller chunks** (500-800): Faster processing, lower memory usage, but may miss broader context
  - **Larger chunks** (1500-2500): Better contextual understanding, but slower processing and higher API costs
  - **Very large chunks** (>3000): May exceed LLM context limits and cause failures

#### 2. Overlap (`--overlap`)
- **Description**: The number of characters overlapping between adjacent chunks
- **Performance Impact**:
  - **Lower overlap** (50-100): Faster processing, fewer chunks overall, but may miss cross-boundary information
  - **Higher overlap** (200-400): Better context preservation, but increases number of chunks and processing time
  - **Optimal ratio**: 10-20% of chunk size strikes a good balance

#### 3. Instruction-Response Pairs Per Chunk (`--pairs`)
- **Description**: Number of instruction-response pairs generated from each text chunk
- **Performance Impact**:
  - **Fewer pairs** (1-2): Faster processing, lower API costs, but less diverse instruction patterns
  - **More pairs** (4-6): Greater dataset diversity, but linearly increases API usage and processing time
  - **API cost**: Each additional pair per chunk increases API costs proportionally

#### 4. Worker Threads (`--workers`)
- **Description**: Number of parallel processes for PDF processing
- **Performance Impact**:
  - **Single worker**: Consistent memory usage, works on any machine, but slower for multiple PDFs
  - **Multiple workers**: Significantly faster for multiple documents, but higher memory usage
  - **System implications**: Optimal value typically equals CPU core count minus 1

### Recommended Configurations

#### Quick Start Configuration (Balanced)
```bash
python main.py --pdf your_document.pdf --chunk_size 1000 --overlap 150 --pairs 3 --workers 2
```
- **Best for**: First-time users, testing the pipeline, balanced speed/quality

#### Economy Configuration (Minimize API costs)
```bash
python main.py --pdf your_document.pdf --chunk_size 1500 --overlap 100 --pairs 2 --workers 1
```
- **Best for**: Large documents, API cost constraints, getting fewer but high-quality pairs

#### High-Quality Configuration (Maximize dataset quality)
```bash
python main.py --pdf your_document.pdf --chunk_size 800 --overlap 200 --pairs 5 --workers 4
```
- **Best for**: Creating comprehensive datasets, documents with complex information

#### Batch Processing Configuration (Maximize throughput)
```bash
python main.py --pdf your_pdf_directory --chunk_size 1200 --overlap 150 --pairs 3 --workers 8
```
- **Best for**: Processing multiple PDFs, systems with 8+ CPU cores

### Performance Benchmarks

| Configuration | Processing Speed | API Calls | Memory Usage | Quality |
|---------------|-----------------|-----------|--------------|---------|
| Quick Start   | Moderate        | Moderate  | Low          | Good    |
| Economy       | Fast            | Low       | Very Low     | Fair    |
| High-Quality  | Slow            | High      | Moderate     | Excellent |
| Batch         | Very Fast       | High      | High         | Good    |

### Hardware Considerations

- **RAM requirements**: Minimum 4GB, recommended 8GB+ for batch processing
- **CPU cores**: More cores = better parallel processing with multiple workers
- **Network**: Stable internet connection required for API calls
- **Storage**: Minimal requirements, primarily for storing output datasets

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google Generative AI API key

### Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/generate_data.git
cd generate_data
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Google API key
```
GOOGLE_API_KEY=your_api_key_here
```

## 💻 Usage

### Command Line Interface

Process a single PDF file:
```bash
python main.py --pdf path/to/your/file.pdf --output output_directory
```

Process multiple PDFs in a directory with parallel workers:
```bash
python main.py --pdf path/to/pdf_directory --output output_directory --workers 4
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--pdf` | Path to PDF file or directory containing PDFs | (required) |
| `--output` | Output directory for generated datasets | "output" |
| `--chunk_size` | Size of text chunks in characters | 1000 |
| `--overlap` | Overlap between chunks in characters | 100 |
| `--pairs` | Number of instruction-response pairs per chunk | 3 |
| `--workers` | Number of worker threads for parallel processing | 1 |
| `--verbose` | Enable verbose logging | False |

### Python API Usage

```python
from src.pdf_processor import PDFProcessor
from src.data_generator import DataGenerator
from src.formatter_dataset import AlpacaFormatter

# Process a PDF file
processor = PDFProcessor(chunk_size=1000, overlap=200)
chunks = processor.process_pdf("path/to/your/document.pdf")

# Generate instruction-response pairs
generator = DataGenerator()
dataset = generator.generate_dataset(chunks, pair_per_chunk=3)

# Format and save the dataset
AlpacaFormatter.format_dataset(dataset, "output/my_dataset.json")

# Advanced: Merge multiple datasets
AlpacaFormatter.merge_datasets(
    ["output/dataset1.json", "output/dataset2.json"], 
    "output/merged_dataset.json"
)

# Convert to JSONL format (for compatibility with some training libraries)
AlpacaFormatter.convert_to_jsonl("output/my_dataset.json")
```

## 📋 Alpaca Format

The generated dataset follows the Alpaca format with entries containing:

- **instruction**: A question or task related to the document content
- **input**: Additional context for the instruction (may be empty)
- **output**: The expected response to the instruction

Example:
```json
{
  "instruction": "Summarize the key points about quantum computing discussed in the text.",
  "input": "",
  "output": "The key points about quantum computing include: 1) It uses quantum bits that can exist in multiple states simultaneously due to superposition, 2) Quantum entanglement allows qubits to be correlated in ways that classical bits cannot, 3) Quantum algorithms can solve certain problems exponentially faster than classical algorithms."
}
```

## 🔧 Performance Optimizations

- **Smart Chunking**: Uses paragraph and sentence boundaries for natural text splitting
- **Parallel Processing**: Processes multiple PDFs simultaneously
- **Retry Logic**: Automatically retries failed API calls with exponential backoff
- **Rate Limiting**: Controls API request frequency to avoid hitting rate limits

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Google Generative AI for powering the instruction-response generation
- PyPDF2 for PDF text extraction capabilities
- Tenacity for retry logic implementation

## 📚 References

1. [GPT-4-LLM Repository](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) - Contains example Alpaca format data used for instruction tuning
2. [Alpaca Data Cleaned](https://github.com/gururise/AlpacaDataCleaned) - Repository addressing common issues in instruction datasets
3. [ML Expert: Alpaca Fine-Tuning Guide](https://www.mlexpert.io/blog/alpaca-fine-tuning) - Tutorial on fine-tuning models with Alpaca format data 