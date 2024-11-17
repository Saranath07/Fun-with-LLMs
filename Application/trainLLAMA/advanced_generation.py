from typing import List, Dict


class ProposalGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate_proposal(
        self,
        requirement: Dict[str, str],
        max_length: int = 1024,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        do_sample: bool = True
    ) -> List[str]:
        """Generate proposal with advanced parameters."""
        # Create structured prompt
        prompt = f"""Requirements Overview:
{requirement['overview']}

Technical Requirements:
{requirement['technical_requirements']}

Deliverables:
{requirement['deliverables']}

Timeline:
{requirement['timeline']}

Additional Information:
{requirement['other']}

Generate a formal client proposal based on the above requirements:

Proposal:"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        return [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]