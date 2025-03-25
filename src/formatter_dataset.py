import json
import os
import logging
from typing import Dict, List, Any, Optional


class AlpacaFormatter:
    """
    A utility class for formatting and validating datasets in the Alpaca format.
    
    The Alpaca format consists of entries with 'instruction', 'input', and 'output' fields
    used for instruction-tuning language models.
    """
    
    # Initialize logger
    logger = logging.getLogger(__name__)

    @staticmethod
    def validate_entry(entry: Dict[str, Any]) -> bool:
        """
        Validate if an entry contains all the required keys for the Alpaca format.
        
        Args:
            entry: A dictionary representing a single data entry
            
        Returns:
            bool: True if the entry contains all required keys, False otherwise
        """
        required_keys = ["instruction", "input", "output"]
        
        # Check if all required keys exist
        if not all(key in entry for key in required_keys):
            return False
            
        # Check if values are not empty/None for instruction and output
        if not entry.get("instruction") or not entry.get("output"):
            return False
            
        # Input can be empty, but must exist
        if "input" not in entry:
            return False
            
        return True

    @staticmethod
    def clean_entry(entry: Dict[str, Any]) -> Dict[str, str]:
        """
        Clean and normalize an entry to ensure consistent formatting.
        
        Args:
            entry: A dictionary representing a single data entry
            
        Returns:
            Dict[str, str]: A cleaned entry with consistent string values
        """
        cleaned = {}
        
        # Ensure all values are strings and strip whitespace
        for key in ["instruction", "input", "output"]:
            if key in entry:
                cleaned[key] = str(entry[key]).strip()
            else:
                cleaned[key] = ""
                
        return cleaned

    @classmethod
    def format_dataset(
        cls,
        data: List[Dict[str, Any]], 
        output_path: str, 
        pretty: bool = True,
        backup_existing: bool = True
    ) -> None:
        """
        Filter valid entries and save the dataset to a JSON file.
        
        Args:
            data: List of data entries to be formatted and saved
            output_path: File path where the formatted dataset will be saved
            pretty: Whether to format the JSON with indentation (default: True)
            backup_existing: Whether to create a backup if the output file already exists
            
        Returns:
            None
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Backup existing file if requested
        if backup_existing and os.path.exists(output_path):
            backup_path = f"{output_path}.bak"
            try:
                import shutil
                shutil.copy2(output_path, backup_path)
                cls.logger.info(f"Created backup of existing file at {backup_path}")
            except Exception as e:
                cls.logger.warning(f"Failed to create backup: {str(e)}")
        
        # Validate and clean entries
        valid_entries = []
        skipped_entries = 0
        
        for entry in data:
            if cls.validate_entry(entry):
                valid_entries.append(cls.clean_entry(entry))
            else:
                skipped_entries += 1
                
        if skipped_entries > 0:
            cls.logger.warning(f"Skipped {skipped_entries} invalid entries")
            
        # Check if we have any valid entries
        if not valid_entries:
            cls.logger.error("No valid entries found in the dataset")
            return
            
        # Set JSON formatting options
        json_kwargs = {"ensure_ascii": False}
        if pretty:
            json_kwargs["indent"] = 2
            
        # Save to file with error handling
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(valid_entries, f, **json_kwargs)
            cls.logger.info(f"Dataset was saved to {output_path} with {len(valid_entries)} valid entries")
        except Exception as e:
            cls.logger.error(f"Error saving dataset: {str(e)}")
            
    @classmethod
    def merge_datasets(cls, input_paths: List[str], output_path: str) -> None:
        """
        Merge multiple Alpaca format datasets into a single file.
        
        Args:
            input_paths: List of paths to input dataset files
            output_path: Path to save the merged dataset
            
        Returns:
            None
        """
        merged_data = []
        
        for path in input_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if not isinstance(data, list):
                    cls.logger.warning(f"File {path} does not contain a JSON array, skipping")
                    continue
                    
                valid_count = 0
                for entry in data:
                    if cls.validate_entry(entry):
                        merged_data.append(cls.clean_entry(entry))
                        valid_count += 1
                        
                cls.logger.info(f"Added {valid_count} entries from {path}")
                
            except Exception as e:
                cls.logger.error(f"Error processing {path}: {str(e)}")
                
        if merged_data:
            cls.format_dataset(merged_data, output_path)
        else:
            cls.logger.error("No valid entries found across input files")
            
    @staticmethod
    def convert_to_jsonl(input_path: str, output_path: Optional[str] = None) -> None:
        """
        Convert a JSON array file to JSONL format (one JSON object per line).
        
        Args:
            input_path: Path to the input JSON file
            output_path: Path for the output JSONL file. If None, 
                         replaces the extension of input_path with .jsonl
                         
        Returns:
            None
        """
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + '.jsonl'
            
        try:
            # Read JSON array
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                raise ValueError(f"Input file does not contain a JSON array")
                
            # Write as JSONL
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    
            logging.info(f"Converted {input_path} to JSONL format at {output_path}")
            
        except Exception as e:
            logging.error(f"Error converting to JSONL: {str(e)}")
            raise