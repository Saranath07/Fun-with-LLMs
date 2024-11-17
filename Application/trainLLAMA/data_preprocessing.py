import re
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict, Tuple
import logging
import os

class DataPreprocessor:
    def __init__(self):
        nltk.download('punkt')
        self.logger = logging.getLogger(__name__)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def structure_requirements(self, text: str) -> Dict[str, str]:
        """Structure requirements into sections."""
        sections = {
            'overview': '',
            'technical_requirements': '',
            'deliverables': '',
            'timeline': '',
            'other': ''
        }
        
        # Simple section detection based on keywords
        sentences = sent_tokenize(text)
        current_section = 'other'
        
        for sentence in sentences:
            lower_sent = sentence.lower()
            if any(kw in lower_sent for kw in ['overview', 'introduction', 'background']):
                current_section = 'overview'
            elif any(kw in lower_sent for kw in ['technical', 'technology', 'system']):
                current_section = 'technical_requirements'
            elif any(kw in lower_sent for kw in ['deliverable', 'output', 'result']):
                current_section = 'deliverables'
            elif any(kw in lower_sent for kw in ['timeline', 'schedule', 'deadline']):
                current_section = 'timeline'
            
            sections[current_section] += sentence + ' '
        
        return {k: v.strip() for k, v in sections.items()}
    
    def process_files(self, requirements_dir: str, proposals_dir: str) -> pd.DataFrame:
        """Process all files and return structured dataset."""
        processed_data = []
        
        for req_file, prop_file in zip(os.listdir(requirements_dir), os.listdir(proposals_dir)):
            try:
                # Read files
                with open(os.path.join(requirements_dir, req_file), 'r') as f:
                    req_text = f.read()
                with open(os.path.join(proposals_dir, prop_file), 'r') as f:
                    prop_text = f.read()
                
                # Clean texts
                req_text = self.clean_text(req_text)
                prop_text = self.clean_text(prop_text)
                
                # Structure requirements
                structured_req = self.structure_requirements(req_text)
                
                processed_data.append({
                    'req_file': req_file,
                    'prop_file': prop_file,
                    'requirements': structured_req,
                    'proposal': prop_text
                })
                
            except Exception as e:
                self.logger.error(f"Error processing files {req_file}, {prop_file}: {str(e)}")
                
        return pd.DataFrame(processed_data)