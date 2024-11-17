from rouge_score import rouge_scorer
from bert_score import score
import torch
from typing import Dict, List
import numpy as np

class ValidationMetrics:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scores = []
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(pred, ref)
            scores.append({
                'rouge1': score['rouge1'].fmeasure,
                'rouge2': score['rouge2'].fmeasure,
                'rougeL': score['rougeL'].fmeasure
            })
        
        # Average scores
        avg_scores = {
            metric: np.mean([s[metric] for s in scores])
            for metric in scores[0].keys()
        }
        
        return avg_scores
    
    def calculate_bert_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore."""
        P, R, F1 = score(predictions, references, lang='en', verbose=False)
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
    
    def evaluate_proposals(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate all metrics."""
        rouge_scores = self.calculate_rouge(predictions, references)
        bert_scores = self.calculate_bert_score(predictions, references)
        
        return {
            **rouge_scores,
            **bert_scores
        }