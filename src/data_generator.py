import os
import json
import time
import logging
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()


class DataGenerator:
    """
    A class for generating instruction-response pairs in Alpaca format using Google's Generative AI.
    
    This generator uses Gemini 1.5 Pro to create instruction-response pairs from text chunks,
    suitable for fine-tuning language models with instruction following capabilities.
    """

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, 
                 rate_limit_delay: float = 1.0):
        """
        Initialize the data generator with Google API credentials.
        
        Args:
            api_key: Optional API key for Google Generative AI.
                    If None, it will be loaded from GOOGLE_API_KEY environment variable.
            max_retries: Maximum number of retries for API calls on failure
            rate_limit_delay: Delay between API calls in seconds to avoid rate limiting
        """
        self.logger = logging.getLogger(__name__)
        
        # Get API key from args or environment
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key not provided and GOOGLE_API_KEY environment variable not set")
            
        # Configure Generative AI
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-002')
        
        # Set parameters
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((genai.types.generation_types.StopCandidateException, 
                                      ConnectionError, TimeoutError))
    )
    def _generate_content(self, prompt: str) -> str:
        """
        Generate content from the Gemini model with retry logic.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The generated text response
            
        Raises:
            Various exceptions from the genai library after retries are exhausted
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.warning(f"API call failed: {str(e)}, retrying...")
            raise

    def _extract_json_from_response(self, content: str) -> List[Dict[str, str]]:
        """
        Extract valid JSON from model response text.
        
        Args:
            content: Text response from the model
            
        Returns:
            Parsed JSON data as a list of dictionaries
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        # Try different extraction methods
        json_str = content
        
        # Extract from markdown code blocks if present
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.rfind("```")
            json_str = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.rfind("```")
            json_str = content[start:end].strip()
            
        # Clean up common JSON formatting issues
        json_str = json_str.replace('\n', ' ').replace('\r', '')
        
        try:
            data = json.loads(json_str)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            self.logger.debug(f"Raw content: {content}")
            raise

    def generate_alpaca_pairs(self, chunk: str, num_pairs: int = 3) -> List[Dict[str, str]]:
        """
        Generate instruction-response pairs in Alpaca format from a text chunk.
        
        Args:
            chunk: Text content from which to generate instruction-response pairs
            num_pairs: Number of pairs to generate from the chunk
            
        Returns:
            List of dictionaries, each containing 'instruction', 'input', and 'output' keys
        """
        if not chunk or not chunk.strip():
            self.logger.warning("Empty chunk provided, skipping")
            return []
            
        # Truncate extremely long chunks to avoid context length errors
        max_chunk_length = 12000  # Adjust based on model capabilities
        if len(chunk) > max_chunk_length:
            self.logger.warning(f"Chunk too long ({len(chunk)} chars), truncating to {max_chunk_length}")
            chunk = chunk[:max_chunk_length] + "..."
        
        prompt = f"""
        I have a chunk of text from a PDF document that I want to convert into instruction-response pairs for finetuning an Alpaca format. 

        Here's the chunk text:
        '''
        {chunk}
        '''
        For each chunk, generate {num_pairs} distinct and relevant instruction-response pairs. 
        The instructions should be clear and concise, requesting specific information or tasks related to the chunk's content.
        Each pair should have: 
        1. "instruction" field: a question or task related to the content
        2. "input" field: could be empty, or provide context for instruction 
        3. "output" field: the expected response

        Make the instructions diverse: include questions about specific facts, requests for explanations, summaries, or analyses.
        Make sure each instruction is specific to the content and requires knowledge from the text to answer correctly.

        Return your response as a valid JSON array with {num_pairs} objects, each having "instruction", "input", "output" keys.
        """

        try:
            # Generate content with retry logic
            content = self._generate_content(prompt)
            
            # Add rate limiting delay
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
                
            # Extract and validate JSON
            data = self._extract_json_from_response(content)
            
            # Validate entries
            valid_entries = []
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                    
                if all(key in entry for key in ["instruction", "input", "output"]):
                    # Ensure all values are strings
                    entry["instruction"] = str(entry["instruction"])
                    entry["input"] = str(entry["input"])
                    entry["output"] = str(entry["output"])
                    valid_entries.append(entry)
                else:
                    self.logger.warning(f"Invalid entry missing required keys: {entry}")
            
            if len(valid_entries) < len(data):
                self.logger.warning(f"Only {len(valid_entries)} of {len(data)} entries were valid")
                
            return valid_entries

        except Exception as e:
            self.logger.error(f"Error generating pairs: {str(e)}")
            return []
    
    def generate_dataset(self, chunks: List[str], pair_per_chunk: int = 3) -> List[Dict[str, str]]:
        """
        Generate a complete dataset of instruction-response pairs from multiple text chunks.
        
        Args:
            chunks: List of text chunks from which to generate pairs
            pair_per_chunk: Number of instruction-response pairs to generate per chunk
            
        Returns:
            A list of dictionaries containing instruction-response pairs in Alpaca format
        """
        dataset = []
        failed_chunks = 0
        
        # Calculate total target pairs
        total_target_pairs = len(chunks) * pair_per_chunk
        
        # Use tqdm for progress tracking
        for i, chunk in enumerate(tqdm(chunks, desc='Generating instruction-response pairs')):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            pairs = self.generate_alpaca_pairs(chunk, pair_per_chunk)
            
            if pairs:
                dataset.extend(pairs)
                self.logger.info(f"Generated {len(pairs)} pairs from chunk {i+1}")
            else:
                failed_chunks += 1
                self.logger.warning(f"Failed to generate pairs from chunk {i+1}")
        
        # Log generation statistics
        total_pairs = len(dataset)
        success_rate = total_pairs / total_target_pairs if total_target_pairs > 0 else 0
        
        self.logger.info(f"Dataset generation complete. Generated {total_pairs} pairs from {len(chunks)} chunks.")
        self.logger.info(f"Success rate: {success_rate:.1%} ({failed_chunks} chunks failed)")
        
        return dataset






        