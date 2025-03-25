import os
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional

from src.pdf_processor import PDFProcessor
from src.data_generator import DataGenerator
from src.formatter_dataset import AlpacaFormatter


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging settings for the application.
    
    Args:
        verbose: If True, sets logging level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Command line arguments
        
    Raises:
        ValueError: If any arguments have invalid values
    """
    if args.chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if args.overlap < 0:
        raise ValueError("Overlap must be non-negative")
    if args.pairs <= 0:
        raise ValueError("Number of pairs must be positive")


def get_pdf_files(pdf_path: str) -> List[str]:
    """
    Get list of PDF files from path (single file or directory).
    
    Args:
        pdf_path: Path to PDF file or directory containing PDF files
        
    Returns:
        List of absolute paths to PDF files
        
    Raises:
        FileNotFoundError: If the path doesn't exist
    """
    if os.path.isfile(pdf_path):
        return [os.path.abspath(pdf_path)]
    elif os.path.isdir(pdf_path):
        pdf_files = [
            os.path.abspath(os.path.join(pdf_path, f)) 
            for f in os.listdir(pdf_path) 
            if f.lower().endswith('.pdf')
        ]
        if not pdf_files:
            logging.warning(f"No PDF files found in directory: {pdf_path}")
        return pdf_files
    else:
        raise FileNotFoundError(f"Path not found: {pdf_path}")


def process_pdf(
    pdf_path: str, 
    output_dir: str, 
    pdf_processor: PDFProcessor, 
    data_generator: DataGenerator,
    pairs_per_chunk: int
) -> Optional[str]:
    """
    Process a single PDF file and generate instruction-response pairs.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the output
        pdf_processor: PDFProcessor instance
        data_generator: DataGenerator instance
        pairs_per_chunk: Number of instruction-response pairs to generate per chunk
        
    Returns:
        Path to the output file if successful, None otherwise
    """
    try:
        logging.info(f"Processing PDF: {pdf_path}")
        
        # Create output filename with safe timestamp format
        filename = os.path.basename(pdf_path).replace('.pdf', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{filename}_{timestamp}.json")
        
        # Extract text chunks
        chunks = pdf_processor.process_pdf(pdf_path)
        logging.info(f"Extracted {len(chunks)} chunks from PDF")
        
        if not chunks:
            logging.warning(f"No text chunks extracted from {pdf_path}")
            return None
            
        # Generate instruction-response pairs
        dataset = data_generator.generate_dataset(chunks, pair_per_chunk=pairs_per_chunk)
        
        if not dataset:
            logging.warning(f"No instruction-response pairs generated for {pdf_path}")
            return None
            
        # Format and save dataset
        AlpacaFormatter.format_dataset(dataset, output_path)
        return output_path
        
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {str(e)}")
        return None


def main():
    """Main function to run the PDF to Alpaca format conversion."""
    parser = argparse.ArgumentParser(description="Convert PDF documents to Alpaca format for LLM fine-tuning")
    parser.add_argument("--pdf", type=str, required=True, help="Path to PDF file or directory containing PDF files")
    parser.add_argument("--output", type=str, default="output", help="Output directory for generated datasets")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of text chunks in characters")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap between chunks in characters")
    parser.add_argument("--pairs", type=int, default=3, help="Number of instruction-response pairs per chunk")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads for parallel processing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Validate arguments
        validate_args(args)
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Initialize processors
        pdf_processor = PDFProcessor(chunk_size=args.chunk_size, overlap=args.overlap)
        data_generator = DataGenerator()
        
        # Get list of PDF files
        pdf_files = get_pdf_files(args.pdf)
        
        if not pdf_files:
            logging.error("No PDF files to process")
            return
            
        logging.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process PDFs in parallel if multiple workers are specified
        successful = 0
        failed = 0
        
        if args.workers > 1 and len(pdf_files) > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                results = list(executor.map(
                    lambda pdf: process_pdf(pdf, args.output, pdf_processor, data_generator, args.pairs),
                    pdf_files
                ))
                successful = sum(1 for r in results if r is not None)
                failed = len(pdf_files) - successful
        else:
            # Process sequentially
            results = [process_pdf(pdf, args.output, pdf_processor, data_generator, args.pairs) 
                      for pdf in pdf_files]
            successful = sum(1 for r in results if r is not None)
            failed = len(pdf_files) - successful
        
        logging.info(f"Processing complete: {successful} successful, {failed} failed")
        
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")


if __name__ == "__main__":
    main()




