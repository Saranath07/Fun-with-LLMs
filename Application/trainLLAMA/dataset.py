import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List, Union
from transformers import PreTrainedTokenizer

class ProposalDataset(Dataset):
    def __init__(
        self,
        data: Union[pd.DataFrame, List[Dict]],
        max_length: int,
        tokenizer: PreTrainedTokenizer = None,
    ):
        """
        Initialize the ProposalDataset.
        
        Args:
            data: DataFrame or list of dictionaries containing 'requirements' and 'proposal'
            max_length: Maximum sequence length for tokenization
            tokenizer: Tokenizer for encoding texts
        """
        self.data = data
        self.max_length = max_length
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        return len(self.data)
    
    def _create_prompt(self, requirements: Dict[str, str]) -> str:
        """Create formatted prompt from requirements."""
        prompt = f"""Requirements Overview:
{requirements['overview']}

Technical Requirements:
{requirements['technical_requirements']}

Deliverables:
{requirements['deliverables']}

Timeline:
{requirements['timeline']}

Additional Information:
{requirements['other']}

Generate a formal client proposal based on the above requirements:

Proposal:"""
        return prompt
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        item = self.data.iloc[idx] if isinstance(self.data, pd.DataFrame) else self.data[idx]
        
        # Create prompt from requirements
        prompt = self._create_prompt(item['requirements'])
        
        # Combine prompt and proposal
        full_text = f"{prompt}{item['proposal']}"
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'][0],
            'attention_mask': encoded['attention_mask'][0],
            'labels': encoded['input_ids'][0].clone()
        }