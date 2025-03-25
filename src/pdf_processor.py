import os
import logging
import PyPDF2
from tqdm import tqdm
from typing import List, Optional


class PDFProcessor:
    """
    A class for processing PDF files and extracting text content in manageable chunks.
    
    This processor extracts text from PDFs and splits it into overlapping chunks
    for further processing, such as generating training data for language models.
    """
    
    def __init__(self, chunk_size: int, overlap: int):
        """
        Initialize the PDF processor with chunking parameters.
        
        Args:
            chunk_size: The target size of each text chunk in characters
            overlap: The number of characters to overlap between adjacent chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = logging.getLogger(__name__)

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract all text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: The extracted text content from all pages
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            PyPDF2.errors.PdfReadError: If the PDF file is corrupted or invalid
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                if num_pages == 0:
                    self.logger.warning(f"PDF file has no pages: {pdf_path}")
                    return text
                
                # Use tqdm for page extraction progress
                for page_num in tqdm(range(num_pages), desc="Extracting pages", leave=False):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:  # Only add non-empty pages
                            text += page_text + '\n\n'
                        else:
                            self.logger.debug(f"Empty text on page {page_num+1}")
                    except Exception as e:
                        self.logger.warning(f"Error extracting text from page {page_num+1}: {str(e)}")
                        # Continue with other pages
        except PyPDF2.errors.PdfReadError as e:
            self.logger.error(f"PDF read error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise
        
        return text
    
    def chunk_text(self, text: str) -> List[str]: 
        """
        Split text into overlapping chunks with intelligent boundaries.
        
        The method attempts to split at natural text boundaries like paragraph breaks
        or sentence endings to preserve context and readability.
        
        Args:
            text: The text to be chunked
            
        Returns:
            List[str]: A list of text chunks
        """
        if not text.strip():
            self.logger.warning("Empty text provided for chunking")
            return []
            
        chunks = []
        start = 0
        total_length = len(text)
        
        # Use tqdm for chunking progress
        with tqdm(total=total_length, desc="Chunking text", leave=False) as pbar:
            while start < total_length:
                # Update progress bar
                pbar.update(min(self.chunk_size, total_length - start))
                
                # Calculate end position
                end = min(start + self.chunk_size, total_length)
                
                # Try to find natural break points
                if end < total_length:
                    # First try paragraph breaks (most natural)
                    paragraph_break = text.rfind('\n\n', start, end)
                    if paragraph_break != -1 and paragraph_break > start + self.chunk_size // 2:
                        end = paragraph_break + 2
                    else:
                        # Next try sentence endings
                        # Look for multiple sentence ending punctuation
                        for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                            sentence_end = text.rfind(punct, start, end)
                            if sentence_end != -1 and sentence_end > start + self.chunk_size // 2:
                                end = sentence_end + len(punct)
                                break
                
                # Extract chunk and add to list
                chunk = text[start:end].strip()
                if chunk:  # Only add non-empty chunks
                    chunks.append(chunk)
                
                # Calculate new start position
                new_start = end - self.overlap
                if new_start <= start:
                    # Avoid infinite loop for very small chunks
                    start += 1
                    self.logger.debug(f"Adjusted start position to avoid infinite loop: {start}")
                else:
                    start = new_start
        
        self.logger.info(f"Text split into {len(chunks)} chunks")
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[str]:
        """
        Process a PDF file by extracting text and splitting into chunks.
        
        This is the main method that combines text extraction and chunking.
        
        Args:
            pdf_path: Path to the PDF file to process
            
        Returns:
            List[str]: A list of text chunks from the PDF
            
        Raises:
            Various exceptions from extract_text method
        """
        self.logger.info(f"Processing PDF: {pdf_path}")
        text = self.extract_text(pdf_path)
        
        if not text.strip():
            self.logger.warning(f"No text extracted from PDF: {pdf_path}")
            return []
            
        self.logger.info(f"Extracted {len(text)} characters from PDF")
        chunks = self.chunk_text(text)
        
        # Log statistics about chunks
        if chunks:
            avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
            self.logger.info(f"Average chunk size: {avg_chunk_size:.1f} characters")
            
        return chunks